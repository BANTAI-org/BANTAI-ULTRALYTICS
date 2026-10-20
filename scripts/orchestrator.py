"""
Pipeline Orchestrator - Manages primary and backup ML pipelines.
Monitors stream health and automatically activates backup pipelines on failure.
"""

import os
import sys
import time
import logging
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import redis
import docker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"


@dataclass
class PipelineInstance:
    """Represents a pipeline instance."""
    pipeline_id: str
    pipeline_type: str  # 'primary' or 'backup'
    stream_url: str
    container_id: Optional[str] = None
    state: PipelineState = PipelineState.STOPPED
    start_time: Optional[float] = None
    failure_count: int = 0


@dataclass
class StreamConfig:
    """Stream configuration."""
    stream_id: str
    primary_url: str
    backup_url: str
    output_url: str


class PipelineOrchestrator:
    """Orchestrates ML pipelines with automatic failover."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            decode_responses=True
        )
        
        # Stream configurations
        self.streams = [
            StreamConfig(
                stream_id='stream-1',
                primary_url=os.getenv('STREAM_1_PRIMARY_URL', 'rtsp://camera1:8554/stream'),
                backup_url=os.getenv('STREAM_1_BACKUP_URL', 'rtsp://camera1-backup:8554/stream'),
                output_url=os.getenv('STREAM_1_OUTPUT_URL', 'rtsp://output:8554/stream1')
            ),
            StreamConfig(
                stream_id='stream-2',
                primary_url=os.getenv('STREAM_2_PRIMARY_URL', 'rtsp://camera2:8554/stream'),
                backup_url=os.getenv('STREAM_2_BACKUP_URL', 'rtsp://camera2-backup:8554/stream'),
                output_url=os.getenv('STREAM_2_OUTPUT_URL', 'rtsp://output:8554/stream2')
            )
        ]
        
        # Pipeline instances
        self.pipelines: Dict[str, PipelineInstance] = {}
        
        # Initialize primary pipelines
        for stream in self.streams:
            pipeline_id = f"primary-{stream.stream_id}"
            self.pipelines[pipeline_id] = PipelineInstance(
                pipeline_id=pipeline_id,
                pipeline_type='primary',
                stream_url=stream.primary_url
            )
            
        # Initialize backup pipelines
        for stream in self.streams:
            pipeline_id = f"backup-{stream.stream_id}"
            self.pipelines[pipeline_id] = PipelineInstance(
                pipeline_id=pipeline_id,
                pipeline_type='backup',
                stream_url=stream.backup_url
            )
            
        self.running = False
        
    def start_pipeline(self, pipeline: PipelineInstance, output_url: str) -> bool:
        """Start a pipeline container."""
        try:
            logger.info(f"Starting pipeline {pipeline.pipeline_id}")
            
            environment = {
                'PIPELINE_ID': pipeline.pipeline_id,
                'PIPELINE_TYPE': pipeline.pipeline_type,
                'STREAM_URL': pipeline.stream_url,
                'OUTPUT_STREAM_URL': output_url,
                'MODEL_PATH': '/app/weights/yolo26s.pt',
                'REDIS_HOST': os.getenv('REDIS_HOST', 'redis'),
                'REDIS_PORT': os.getenv('REDIS_PORT', '6379'),
                'HEALTH_CHECK_INTERVAL': '10',
                'STREAM_TIMEOUT': '30',
                'ENABLE_TRACKING': os.getenv('ENABLE_TRACKING', 'true'),
                'TRACK_THRESH': os.getenv('TRACK_THRESH', '0.5'),
                'TRACK_BUFFER': os.getenv('TRACK_BUFFER', '30'),
                'MATCH_THRESH': os.getenv('MATCH_THRESH', '0.8')
            }
            
            container = self.docker_client.containers.run(
                'bantai-ml-pipeline:latest',
                name=pipeline.pipeline_id,
                environment=environment,
                network='bantai-network',
                detach=True,
                restart_policy={'Name': 'unless-stopped'},
                runtime='nvidia' if os.getenv('USE_GPU', 'true').lower() == 'true' else None
            )
            
            pipeline.container_id = container.id
            pipeline.state = PipelineState.STARTING
            pipeline.start_time = time.time()
            
            logger.info(f"Pipeline {pipeline.pipeline_id} started (container: {container.short_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline {pipeline.pipeline_id}: {e}")
            pipeline.state = PipelineState.FAILED
            pipeline.failure_count += 1
            return False
            
    def stop_pipeline(self, pipeline: PipelineInstance):
        """Stop a pipeline container."""
        try:
            if pipeline.container_id:
                logger.info(f"Stopping pipeline {pipeline.pipeline_id}")
                container = self.docker_client.containers.get(pipeline.container_id)
                container.stop(timeout=10)
                container.remove()
                pipeline.container_id = None
                
            pipeline.state = PipelineState.STOPPED
            logger.info(f"Pipeline {pipeline.pipeline_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline {pipeline.pipeline_id}: {e}")
            
    def check_pipeline_health(self, pipeline: PipelineInstance) -> bool:
        """Check if pipeline is healthy."""
        try:
            health_key = f'pipeline:{pipeline.pipeline_id}:health'
            health_data = self.redis_client.hgetall(health_key)
            
            if not health_data:
                return False
                
            status = health_data.get('status', 'unknown')
            last_frame_time = float(health_data.get('last_frame_time', 0))
            
            # Check if recent frame received (within last 30 seconds)
            is_healthy = (
                status == 'healthy' and
                (time.time() - last_frame_time) < 30
            )
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Error checking health for {pipeline.pipeline_id}: {e}")
            return False
            
    def handle_stream_failure(self, primary_pipeline_id: str):
        """Handle primary stream failure by activating backup."""
        logger.warning(f"Handling failure for {primary_pipeline_id}")
        
        # Find corresponding backup pipeline
        stream_id = primary_pipeline_id.replace('primary-', '')
        backup_pipeline_id = f"backup-{stream_id}"
        
        backup_pipeline = self.pipelines.get(backup_pipeline_id)
        if not backup_pipeline:
            logger.error(f"No backup pipeline found for {primary_pipeline_id}")
            return
            
        # Find output URL
        stream_config = next((s for s in self.streams if s.stream_id == stream_id), None)
        if not stream_config:
            logger.error(f"No stream config found for {stream_id}")
            return
            
        # Start backup pipeline if not running
        if backup_pipeline.state != PipelineState.RUNNING:
            logger.info(f"Activating backup pipeline {backup_pipeline_id}")
            self.start_pipeline(backup_pipeline, stream_config.output_url)
        else:
            logger.info(f"Backup pipeline {backup_pipeline_id} already running")
            
    def monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Starting pipeline monitoring")
        
        # Start primary pipelines
        for stream in self.streams:
            pipeline_id = f"primary-{stream.stream_id}"
            pipeline = self.pipelines[pipeline_id]
            self.start_pipeline(pipeline, stream.output_url)
            
        # Wait for pipelines to start
        time.sleep(5)
        
        # Subscribe to failure events
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe('stream:failure', 'pipeline:error')
        
        self.running = True
        
        while self.running:
            try:
                # Check messages from Redis
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    
                    if channel == 'stream:failure':
                        logger.warning(f"Stream failure detected: {data}")
                        self.handle_stream_failure(data)
                    elif channel == 'pipeline:error':
                        logger.error(f"Pipeline error: {data}")
                        
                # Periodic health checks
                for pipeline_id, pipeline in self.pipelines.items():
                    if pipeline.state == PipelineState.STARTING:
                        # Check if started successfully
                        if time.time() - pipeline.start_time > 30:
                            if self.check_pipeline_health(pipeline):
                                pipeline.state = PipelineState.RUNNING
                                logger.info(f"Pipeline {pipeline_id} is now running")
                            else:
                                logger.error(f"Pipeline {pipeline_id} failed to start")
                                pipeline.state = PipelineState.FAILED
                                
                    elif pipeline.state == PipelineState.RUNNING:
                        # Check health
                        if not self.check_pipeline_health(pipeline):
                            logger.warning(f"Pipeline {pipeline_id} is unhealthy")
                            pipeline.state = PipelineState.UNHEALTHY
                            
                            # If primary pipeline is unhealthy, activate backup
                            if pipeline.pipeline_type == 'primary':
                                self.handle_stream_failure(pipeline_id)
                                
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(5)
                
        # Cleanup
        self.shutdown()
        
    def shutdown(self):
        """Shutdown all pipelines."""
        logger.info("Shutting down orchestrator")
        
        for pipeline in self.pipelines.values():
            if pipeline.state != PipelineState.STOPPED:
                self.stop_pipeline(pipeline)
                
        logger.info("All pipelines stopped")


if __name__ == '__main__':
    orchestrator = PipelineOrchestrator()
    orchestrator.monitor_loop()
