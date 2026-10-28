"""
ML Pipeline Runner with H.264 stream processing and health monitoring.
Supports primary and backup pipeline modes with Redis coordination.
"""

import os
import sys
import time
import signal
import logging
import json
import tempfile
import subprocess
import threading
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
import redis
import requests

from bytetrack import ByteTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration from environment."""
    pipeline_id: str
    pipeline_type: str  # 'primary' or 'backup'
    stream_url: str
    model_path: str
    redis_host: str
    redis_port: int
    health_check_interval: int
    stream_timeout: int
    output_stream_url: Optional[str]
    enable_tracking: bool = True
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    # Evidence upload (R3)
    evidence_upload_enabled: bool = False
    evidence_api_base_url: Optional[str] = None
    evidence_upload_token: Optional[str] = None
    incident_id: Optional[str] = None
    evidence_no_detect_timeout: float = 5.0  # seconds
    evidence_pre_buffer_seconds: float = 10.0  # keep buffer before first detection
    evidence_use_nvdec: bool = True  # prefer hardware decode (nvdec)
    use_turn: bool = False
    turn_server: Optional[str] = None
    turn_username: Optional[str] = None
    turn_password: Optional[str] = None


class StreamHealthMonitor:
    """Monitors stream health and reports status to Redis."""
    
    def __init__(self, redis_client: redis.Redis, pipeline_id: str):
        self.redis = redis_client
        self.pipeline_id = pipeline_id
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def update(self, frame_received: bool):
        """Update health status."""
        if frame_received:
            self.last_frame_time = time.time()
            self.frame_count += 1
            
        # Update Redis with health status
        health_data = {
            'last_frame_time': self.last_frame_time,
            'frame_count': self.frame_count,
            'status': 'healthy' if frame_received else 'degraded'
        }
        self.redis.hset(f'pipeline:{self.pipeline_id}:health', mapping=health_data)
        self.redis.expire(f'pipeline:{self.pipeline_id}:health', 30)
        
    def is_healthy(self, timeout: int) -> bool:
        """Check if stream is healthy."""
        return (time.time() - self.last_frame_time) < timeout


class EvidenceUploader:
    """Handle evidence clipping and multipart upload to NestJS/R3."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.enabled = config.evidence_upload_enabled and config.evidence_api_base_url
        self.session = requests.Session()
        if config.evidence_upload_token:
            self.session.headers.update({'Authorization': f'Bearer {config.evidence_upload_token}'})
        
    def initiate_multipart(self, file_name: str, content_type: str = 'video/mp4') -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.config.evidence_api_base_url}/storage/evidence/multipart/initiate"
            params = {}
            if self.config.incident_id:
                params['incidentId'] = self.config.incident_id
            resp = self.session.post(url, json={'fileName': file_name, 'contentType': content_type}, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to initiate multipart upload: {e}")
            return None
    
    def sign_parts(self, key: str, upload_id: str, part_numbers: List[int]) -> Optional[List[Dict[str, Any]]]:
        try:
            url = f"{self.config.evidence_api_base_url}/storage/evidence/multipart/sign-parts"
            resp = self.session.post(url, json={'key': key, 'uploadId': upload_id, 'partNumbers': part_numbers})
            resp.raise_for_status()
            data = resp.json()
            return data.get('urls') or data
        except Exception as e:
            logger.error(f"Failed to sign parts: {e}")
            return None
    
    def complete_multipart(self, key: str, upload_id: str, parts: List[Dict[str, Any]]) -> bool:
        try:
            url = f"{self.config.evidence_api_base_url}/storage/evidence/multipart/complete"
            resp = self.session.post(url, json={'key': key, 'uploadId': upload_id, 'parts': parts})
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to complete multipart upload: {e}")
            return False
    
    def upload_file_multipart(self, file_path: str, file_name: Optional[str] = None) -> bool:
        if not self.enabled:
            logger.warning("Evidence upload disabled or missing config")
            return False
        if not os.path.exists(file_path):
            logger.error(f"Evidence file not found: {file_path}")
            return False
        file_name = file_name or os.path.basename(file_path)
        init = self.initiate_multipart(file_name)
        if not init:
            return False
        upload_id = init.get('uploadId')
        key = init.get('objectKey') or init.get('key')
        if not upload_id or not key:
            logger.error(f"Invalid initiate response: {init}")
            return False
        file_size = os.path.getsize(file_path)
        part_size = 5 * 1024 * 1024  # 5MB
        part_numbers = []
        with open(file_path, 'rb') as f:
            i = 1
            while True:
                chunk = f.read(part_size)
                if not chunk:
                    break
                part_numbers.append(i)
                i += 1
        urls_info = self.sign_parts(key, upload_id, part_numbers)
        if not urls_info or len(urls_info) != len(part_numbers):
            logger.error("Failed to get presigned URLs for all parts")
            return False
        parts = []
        with open(file_path, 'rb') as f:
            for idx, pn in enumerate(part_numbers):
                chunk = f.read(part_size)
                url = urls_info[idx].get('url') if isinstance(urls_info[idx], dict) else urls_info[idx]
                try:
                    resp = requests.put(url, data=chunk)
                    resp.raise_for_status()
                    etag = resp.headers.get('ETag') or resp.headers.get('etag')
                    if etag and etag.startswith('"') and etag.endswith('"'):
                        etag = etag[1:-1]
                    parts.append({'partNumber': pn, 'etag': etag or ''})
                except Exception as e:
                    logger.error(f"Failed to upload part {pn}: {e}")
                    return False
        return self.complete_multipart(key, upload_id, parts)


class MLPipeline:
    """ML inference pipeline with H.264 stream processing."""
    
    def __init__(self, config: PipelineConfig, frame_source=None):
        self.config = config
        self.running = False
        self.model = None
        self.capture = frame_source
        self.writer = None
        self.redis = None
        self.health_monitor = None
        self.tracker = None
        self.evidence_uploader = EvidenceUploader(config) if config.evidence_upload_enabled else None
        # Evidence buffering
        self.frame_buffer: List[Tuple[float, np.ndarray]] = []
        self.first_detection_time: Optional[float] = None
        self.last_detection_time: Optional[float] = None
        self.recording_evidence = False
        self.evidence_clip_path: Optional[str] = None
        self.evidence_writer = None
        self.buffer_max_seconds = max(30.0, config.evidence_pre_buffer_seconds + 60.0)
        
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def _build_capture(self):
        """Open video capture with nvdec/turn support if configured."""
        url = self.config.stream_url
        logger.info(f"Opening stream: {url}")
        if self.config.use_turn and self.config.turn_server:
            logger.info(f"Using TURN server config: {self.config.turn_server}")
        if self.config.evidence_use_nvdec:
            # Try to prefer hardware decode; OpenCV with ffmpeg may use hardware if available
            os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'hwaccel;cuda|hwaccel_device;0')
        if self.capture is not None:
            return self.capture
        self.capture = cv2.VideoCapture(url)
        if not self.capture.isOpened():
            # Fallback: try with default options
            self.capture = cv2.VideoCapture(url)
        return self.capture
    
    def initialize(self):
        """Initialize pipeline components."""
        logger.info(f"Initializing {self.config.pipeline_type} pipeline {self.config.pipeline_id}")
        
        # Connect to Redis
        self.redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            decode_responses=True
        )
        self.redis.ping()
        logger.info("Connected to Redis")
        
        # Initialize health monitor
        self.health_monitor = StreamHealthMonitor(self.redis, self.config.pipeline_id)
        
        # Load ML model
        logger.info(f"Loading model from {self.config.model_path}")
        self.model = YOLO(self.config.model_path)
        logger.info("Model loaded successfully")
        
        # Initialize ByteTracker
        if self.config.enable_tracking:
            logger.info("Initializing ByteTrack tracker")
            self.tracker = ByteTracker(
                track_thresh=self.config.track_thresh,
                track_buffer=self.config.track_buffer,
                match_thresh=self.config.match_thresh,
                frame_rate=30
            )
            logger.info("ByteTrack initialized")
        
        # Open input stream
        self._build_capture()
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open stream: {self.config.stream_url}")
            
        # Get stream properties
        fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or 30
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Stream properties: {width}x{height} @ {fps}fps")
        self.fps = fps
        self.frame_size = (width, height)
        
        # Setup output stream (H.264 encoded)
        if self.config.output_stream_url:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.writer = cv2.VideoWriter(
                self.config.output_stream_url,
                fourcc,
                fps,
                (width, height)
            )
            logger.info(f"Output stream: {self.config.output_stream_url}")
            
        # Register pipeline in Redis
        self.redis.hset(f'pipeline:{self.config.pipeline_id}', mapping={
            'type': self.config.pipeline_type,
            'status': 'running',
            'stream_url': self.config.stream_url,
            'started_at': time.time()
        })
        
        self.running = True
        logger.info("Pipeline initialized successfully")
        
    def _update_evidence_buffer(self, frame: np.ndarray, has_detection: bool, ts: float):
        if not self.evidence_uploader or not self.evidence_uploader.enabled:
            return
        # Add to buffer (ring buffer)
        self.frame_buffer.append((ts, frame.copy()))
        # Trim buffer
        if self.frame_buffer:
            cutoff = ts - self.buffer_max_seconds
            while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
                self.frame_buffer.pop(0)
        if has_detection:
            self.last_detection_time = ts
            if not self.recording_evidence:
                self.first_detection_time = ts
                self.recording_evidence = True
                logger.info(f"Evidence recording started at {ts}")
                # Start evidence writer from buffered frames if possible
                self._start_evidence_writer(ts)
        else:
            # Check timeout
            if self.recording_evidence and self.last_detection_time:
                if ts - self.last_detection_time >= self.config.evidence_no_detect_timeout:
                    logger.info(f"No detection for {self.config.evidence_no_detect_timeout}s - finalizing evidence clip")
                    self._finalize_evidence_clip()
    
    def _start_evidence_writer(self, start_ts: float):
        try:
            fps = getattr(self, 'fps', 30)
            size = getattr(self, 'frame_size', None)
            if not size and self.capture:
                w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                size = (w, h)
            if not size:
                return
            tmpdir = tempfile.gettempdir()
            fname = f"evidence_{self.config.pipeline_id}_{int(time.time())}.mp4"
            self.evidence_clip_path = os.path.join(tmpdir, fname)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.evidence_writer = cv2.VideoWriter(self.evidence_clip_path, fourcc, fps, size)
            # Write buffered frames from start of detection window
            if self.first_detection_time:
                for fts, fframe in self.frame_buffer:
                    if fts >= self.first_detection_time - 0.01:  # include from first detection start
                        self.evidence_writer.write(fframe)
        except Exception as e:
            logger.error(f"Failed to start evidence writer: {e}")
            self.evidence_writer = None
            self.evidence_clip_path = None
    
    def _write_evidence_frame(self, frame: np.ndarray):
        if self.recording_evidence and self.evidence_writer is not None:
            try:
                self.evidence_writer.write(frame)
            except Exception as e:
                logger.error(f"Failed to write evidence frame: {e}")
    
    def _finalize_evidence_clip(self):
        try:
            if self.evidence_writer:
                self.evidence_writer.release()
                self.evidence_writer = None
            clip_path = self.evidence_clip_path
            self.recording_evidence = False
            self.first_detection_time = None
            self.last_detection_time = None
            self.evidence_clip_path = None
            if clip_path and os.path.exists(clip_path) and self.evidence_uploader:
                logger.info(f"Uploading evidence clip: {clip_path}")
                # Try to upload in background thread would be better, but keep simple
                try:
                    self.evidence_uploader.upload_file_multipart(clip_path)
                except Exception as e:
                    logger.error(f"Evidence upload failed: {e}")
                finally:
                    try:
                        os.remove(clip_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed to finalize evidence clip: {e}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with ML model and ByteTrack tracking."""
        results = self.model(frame, verbose=False)[0]
        
        annotated_frame = frame.copy()
        has_detection = False
        
        if self.config.enable_tracking and len(results.boxes) > 0:
            has_detection = True
            # Prepare detections for ByteTrack: [x1, y1, x2, y2, score, class_id]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf, cls])
                
            detections = np.array(detections)
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Draw tracked objects
            track_data = []
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id
                score = track.score
                cls = track.cls
                
                # Color based on track ID for consistency
                color = self._get_track_color(track_id)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with track ID
                label = f"ID:{track_id} {results.names[cls]} {score:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                
                track_data.append({
                    'track_id': track_id,
                    'class': results.names[cls],
                    'score': float(score),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
            # Store tracking data in Redis
            detection_data = {
                'count': len(tracks),
                'timestamp': time.time(),
                'tracked_objects': str(track_data)
            }
            self.redis.hset(f'pipeline:{self.config.pipeline_id}:detections', mapping=detection_data)
            
        elif len(results.boxes) > 0:
            has_detection = True
            # No tracking - just draw detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{results.names[cls]} {conf:.2f}"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                
            # Store detections in Redis
            detection_data = {
                'count': len(results.boxes),
                'timestamp': time.time(),
                'classes': ','.join([results.names[int(b.cls[0])] for b in results.boxes])
            }
            self.redis.hset(f'pipeline:{self.config.pipeline_id}:detections', mapping=detection_data)
        else:
            has_detection = False
            detection_data = {
                'count': 0,
                'timestamp': time.time(),
                'classes': ''
            }
            self.redis.hset(f'pipeline:{self.config.pipeline_id}:detections', mapping=detection_data)
        
        # Evidence handling
        try:
            ts = time.time()
            self._update_evidence_buffer(frame, has_detection, ts)
            self._write_evidence_frame(frame)
        except Exception as e:
            logger.error(f"Evidence handling error: {e}")
        
        return annotated_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        # Generate color based on track ID
        np.random.seed(track_id)
        color = tuple(map(int, np.random.randint(50, 255, 3)))
        return color
        
    def run(self):
        """Main pipeline loop."""
        try:
            self.initialize()
            
            frame_count = 0
            last_health_check = time.time()
            
            while self.running:
                ret, frame = self.capture.read()
                
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    self.health_monitor.update(False)
                    
                    # Check if stream is down
                    if not self.health_monitor.is_healthy(self.config.stream_timeout):
                        logger.error("Stream timeout detected")
                        self.redis.publish('stream:failure', self.config.pipeline_id)
                        break
                        
                    time.sleep(0.1)
                    continue
                    
                # Update health status
                self.health_monitor.update(True)
                
                # Process frame with ML model
                processed_frame = self.process_frame(frame)
                
                # Write to output stream
                if self.writer:
                    self.writer.write(processed_frame)
                    
                frame_count += 1
                
                # Periodic health check
                if time.time() - last_health_check > self.config.health_check_interval:
                    logger.info(f"Processed {frame_count} frames, stream healthy")
                    last_health_check = time.time()
                    
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.redis.publish('pipeline:error', f"{self.config.pipeline_id}:{str(e)}")
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up pipeline resources")
        
        if self.capture:
            self.capture.release()
            
        if self.writer:
            self.writer.release()
            
        if self.redis:
            self.redis.hset(f'pipeline:{self.config.pipeline_id}', 'status', 'stopped')
            
        logger.info("Pipeline stopped")


def load_config_from_env() -> PipelineConfig:
    """Load configuration from environment variables."""
    return PipelineConfig(
        pipeline_id=os.getenv('PIPELINE_ID', 'pipeline-1'),
        pipeline_type=os.getenv('PIPELINE_TYPE', 'primary'),
        stream_url=os.getenv('STREAM_URL', 'rtsp://localhost:8554/stream'),
        model_path=os.getenv('MODEL_PATH', '/app/weights/best.pt'),
        redis_host=os.getenv('REDIS_HOST', 'redis'),
        redis_port=int(os.getenv('REDIS_PORT', '6379')),
        health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '10')),
        stream_timeout=int(os.getenv('STREAM_TIMEOUT', '30')),
        output_stream_url=os.getenv('OUTPUT_STREAM_URL'),
        enable_tracking=os.getenv('ENABLE_TRACKING', 'true').lower() == 'true',
        track_thresh=float(os.getenv('TRACK_THRESH', '0.5')),
        track_buffer=int(os.getenv('TRACK_BUFFER', '30')),
        match_thresh=float(os.getenv('MATCH_THRESH', '0.8')),
        evidence_upload_enabled=os.getenv('EVIDENCE_UPLOAD_ENABLED', 'false').lower() == 'true',
        evidence_api_base_url=os.getenv('EVIDENCE_API_BASE_URL') or os.getenv('NESTJS_API_URL') or os.getenv('API_BASE_URL'),
        evidence_upload_token=os.getenv('EVIDENCE_UPLOAD_TOKEN') or os.getenv('API_TOKEN'),
        incident_id=os.getenv('INCIDENT_ID'),
        evidence_no_detect_timeout=float(os.getenv('EVIDENCE_NO_DETECT_TIMEOUT', '5.0')),
        evidence_pre_buffer_seconds=float(os.getenv('EVIDENCE_PRE_BUFFER_SECONDS', '10.0')),
        evidence_use_nvdec=os.getenv('EVIDENCE_USE_NVDEC', 'true').lower() == 'true',
        use_turn=os.getenv('USE_TURN', 'false').lower() == 'true',
        turn_server=os.getenv('TURN_SERVER') or os.getenv('TURN_URL'),
        turn_username=os.getenv('TURN_USERNAME'),
        turn_password=os.getenv('TURN_PASSWORD'),
    )


if __name__ == '__main__':
    config = load_config_from_env()
    pipeline = MLPipeline(config)
    pipeline.run()
