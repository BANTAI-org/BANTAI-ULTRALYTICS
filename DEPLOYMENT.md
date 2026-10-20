# Build and Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (optional but recommended)
- NVIDIA Container Toolkit (for GPU support)

### 1. Build the System

```bash
# Clone repository
cd BANTAI-ULTRALYTICS

# Build all Docker images (includes bytecode compilation)
make build-all
```

### 2. Configure

```bash
cd docker
cp .env.example .env
```

Edit `.env` with your camera URLs and tracking settings:

```env
# Primary camera streams
STREAM_1_PRIMARY_URL=rtsp://192.168.1.101:8554/stream
STREAM_2_PRIMARY_URL=rtsp://192.168.1.102:8554/stream

# Backup camera streams (for failover)
STREAM_1_BACKUP_URL=rtsp://192.168.1.201:8554/stream
STREAM_2_BACKUP_URL=rtsp://192.168.1.202:8554/stream

# Output streams (H.264)
STREAM_1_OUTPUT_URL=rtsp://localhost:8554/stream1
STREAM_2_OUTPUT_URL=rtsp://localhost:8554/stream2

# GPU support
USE_GPU=true

# ByteTrack settings
ENABLE_TRACKING=true
TRACK_THRESH=0.5      # Higher = fewer tracks
TRACK_BUFFER=30       # Frames to keep lost tracks
MATCH_THRESH=0.8      # IoU threshold for matching
```

### 3. Deploy

```bash
# Start all services
make start

# View logs
make logs

# Check status
make status
```

## 🐳 Docker Images

### ML Pipeline Image
- **Base**: Python 3.11-slim
- **Size**: ~2.5GB
- **Includes**: YOLO, ByteTrack, OpenCV, Redis client
- **Optimization**: Bytecode compiled with `-OO`

### Orchestrator Image
- **Base**: Python 3.11-slim  
- **Size**: ~500MB
- **Includes**: Docker client, Redis client
- **Function**: Manages pipeline lifecycle

### RTSP Server
- **Image**: bluenviron/mediamtx
- **Protocols**: RTSP, WebRTC, HLS
- **Ports**: 8554 (RTSP), 8888 (WebRTC), 8889 (HLS)

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Compose Stack                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐       ┌─────────────┐                │
│  │    Redis     │◄──────┤ Orchestrator│                │
│  │  (State DB)  │       │  (Manager)  │                │
│  └──────────────┘       └─────────────┘                │
│         ▲                       │                        │
│         │                       │ Docker API             │
│         │                       ▼                        │
│         │           ┌────────────────────┐              │
│         │           │ Spawns Pipelines   │              │
│         │           └────────────────────┘              │
│         │                                                │
│         │  Health     ┌──────────────┐                 │
│         └─────────────┤ ML Pipeline 1│                 │
│            Checks      │  (Primary)   │                 │
│                        └──────────────┘                 │
│                                │                         │
│                        ┌──────────────┐                 │
│                        │ ML Pipeline 2│                 │
│                        │  (Primary)   │                 │
│                        └──────────────┘                 │
│                                │                         │
│                        ┌──────────────┐                 │
│                        │ ML Pipeline 3│ (Backup)        │
│                        │  (On Demand) │                 │
│                        └──────────────┘                 │
│                                │                         │
│                        ┌──────────────┐                 │
│                        │ ML Pipeline 4│ (Backup)        │
│                        │  (On Demand) │                 │
│                        └──────────────┘                 │
│                                │                         │
│                        ┌──────────────┐                 │
│                        │ RTSP Server  │                 │
│                        │   (Output)   │                 │
│                        └──────────────┘                 │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Management Commands

```bash
# Build images
make build-pipeline       # ML pipeline only
make build-orchestrator   # Orchestrator only
make build-all           # Everything

# Lifecycle
make start               # Start system
make stop                # Stop system
make restart             # Restart system

# Monitoring
make logs                # All logs
make logs-orchestrator   # Orchestrator logs
make logs-redis          # Redis logs
make status              # Container status
make monitor             # Pipeline health

# Cleanup
make clean               # Remove everything
```

## 📈 Monitoring

### Redis Keys

```bash
# Pipeline health
pipeline:{pipeline_id}:health
  - last_frame_time
  - frame_count
  - status

# Tracking data
pipeline:{pipeline_id}:detections
  - count
  - timestamp
  - tracked_objects

# Pipeline metadata
pipeline:{pipeline_id}
  - type (primary/backup)
  - status
  - stream_url
  - started_at
```

### Health Checks

```bash
# Enter Redis CLI
docker exec -it bantai-redis redis-cli

# Check all pipelines
> KEYS pipeline:*

# View pipeline health
> HGETALL pipeline:primary-stream-1:health

# View tracking data
> HGETALL pipeline:primary-stream-1:detections
```

## 🎯 Troubleshooting

### Pipelines won't start

```bash
# Check orchestrator logs
docker logs bantai-orchestrator

# Check pipeline image exists
docker images | grep bantai-ml-pipeline

# Rebuild if needed
make build-pipeline
```

### Stream connection failed

```bash
# Test stream manually
ffmpeg -i rtsp://your-camera-url -frames:v 1 test.jpg

# Check network from container
docker exec bantai-orchestrator ping camera-hostname
```

### GPU not detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check docker-compose GPU config
grep -A5 "runtime:" docker/docker-compose.yml
```

### High memory usage

```bash
# Add resource limits to docker-compose.yml
services:
  orchestrator:
    deploy:
      resources:
        limits:
          memory: 500M
```

## 🔒 Production Hardening

### 1. Enable Redis Authentication

```yaml
# docker-compose.yml
services:
  redis:
    command: redis-server --requirepass YOUR_STRONG_PASSWORD
    environment:
      - REDIS_PASSWORD=YOUR_STRONG_PASSWORD
```

### 2. Add TLS for RTSP

```yaml
# mediamtx.yml
encryption: "strict"
serverCert: /certs/server.crt
serverKey: /certs/server.key
```

### 3. Set Resource Limits

```yaml
services:
  ml-pipeline:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 4. Configure Logging

```yaml
services:
  orchestrator:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 5. Network Isolation

```yaml
networks:
  bantai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

## 📦 Bytecode Compilation

The system automatically compiles Python to bytecode during Docker build:

```dockerfile
# In Dockerfile.ml-pipeline
RUN python -m compileall -b /app/src /app/scripts && \
    find /app -type f -name '*.py' -delete && \
    find /app -type f -name '*.pyc' -exec rename 's/\.pyc$/\.py/' {} \;
```

Benefits:
- **30-40% faster startup**
- **Smaller images** (no source comments/docstrings)
- **Protected code** (harder to reverse engineer)

Manual compilation:
```bash
python scripts/compile_bytecode.py
```

## 🌐 Accessing Output Streams

### RTSP (VLC, ffplay)
```bash
rtsp://localhost:8554/stream1
rtsp://localhost:8554/stream2
```

### WebRTC (Browser)
```
http://localhost:8889
```

### HLS (Browser, mobile)
```
http://localhost:8888/stream1/index.m3u8
http://localhost:8888/stream2/index.m3u8
```

## 📊 Performance Tuning

### Increase Throughput
```env
# Reduce model input size in pipeline_runner.py
# Trade accuracy for speed
```

### Reduce Latency
```yaml
# mediamtx.yml
hlsSegmentDuration: 1s
hlsSegmentCount: 3
```

### Optimize Tracking
```env
# Faster tracking, more ID switches
TRACK_BUFFER=15
MATCH_THRESH=0.7

# Slower tracking, stable IDs
TRACK_BUFFER=60
MATCH_THRESH=0.9
```

## 🔄 Updates & Rollback

### Update Images
```bash
# Pull new code
git pull

# Rebuild
make build-all

# Rolling update (zero downtime)
docker-compose up -d --no-deps --build orchestrator
```

### Rollback
```bash
# Tag before deploying
docker tag bantai-ml-pipeline:latest bantai-ml-pipeline:backup

# Rollback if needed
docker tag bantai-ml-pipeline:backup bantai-ml-pipeline:latest
make restart
```

---

For more details, see:
- [README-PIPELINES.md](./README-PIPELINES.md) - System overview
- [README-BYTETRACK.md](./README-BYTETRACK.md) - Tracking details
