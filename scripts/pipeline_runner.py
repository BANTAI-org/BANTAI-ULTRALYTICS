"""
ML Pipeline Runner with H.264 stream processing and health monitoring.
Supports primary and backup pipeline modes with Redis coordination.
"""

import os
import sys
import time
import signal
import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import redis

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


class MLPipeline:
    """ML inference pipeline with H.264 stream processing."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.running = False
        self.model = None
        self.capture = None
        self.writer = None
        self.redis = None
        self.health_monitor = None
        self.tracker = None
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
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
        logger.info(f"Opening stream: {self.config.stream_url}")
        self.capture = cv2.VideoCapture(self.config.stream_url)
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open stream: {self.config.stream_url}")
            
        # Get stream properties
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Stream properties: {width}x{height} @ {fps}fps")
        
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
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with ML model and ByteTrack tracking."""
        results = self.model(frame, verbose=False)[0]
        
        annotated_frame = frame.copy()
        
        if self.config.enable_tracking and len(results.boxes) > 0:
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
            
        else:
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
        model_path=os.getenv('MODEL_PATH', '/app/weights/yolo26s.pt'),
        redis_host=os.getenv('REDIS_HOST', 'redis'),
        redis_port=int(os.getenv('REDIS_PORT', '6379')),
        health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '10')),
        stream_timeout=int(os.getenv('STREAM_TIMEOUT', '30')),
        output_stream_url=os.getenv('OUTPUT_STREAM_URL'),
        enable_tracking=os.getenv('ENABLE_TRACKING', 'true').lower() == 'true',
        track_thresh=float(os.getenv('TRACK_THRESH', '0.5')),
        track_buffer=int(os.getenv('TRACK_BUFFER', '30')),
        match_thresh=float(os.getenv('MATCH_THRESH', '0.8'))
    )


if __name__ == '__main__':
    config = load_config_from_env()
    pipeline = MLPipeline(config)
    pipeline.run()
