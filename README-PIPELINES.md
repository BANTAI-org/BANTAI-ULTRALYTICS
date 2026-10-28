  # BANTAI ML Runtime Pipelines

Resilient machine learning pipeline system with automatic failover for continuous H.264 video stream processing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                     │
│              (Monitors & Manages Pipelines)                  │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
    ┌─────────▼─────────┐       ┌────────▼────────┐
    │   Primary ML 1    │       │   Primary ML 2   │
    │   (Stream 1)      │       │   (Stream 2)     │
    └─────────┬─────────┘       └────────┬─────────┘
              │                           │
              │  Network Failure Detected │
              │                           │
    ┌─────────▼─────────┐       ┌────────▼────────┐
    │   Backup ML 1     │       │   Backup ML 2    │
    │ (Backup Stream 1) │       │ (Backup Stream 2)│
    └─────────┬─────────┘       └────────┬─────────┘
              │                           │
              └───────────┬───────────────┘
                          │
                   ┌──────▼──────┐
                   │ H.264 Output│
                   │   Streams   │
                   └─────────────┘
```

## Features

- **2 Primary ML Pipelines**: Process main video streams continuously
- **2 Backup Pipelines**: Automatically activate when primary streams fail
- **Stream Health Monitoring**: Real-time detection of network failures
- **H.264 Encoding**: Optimized video output for streaming
- **Redis Coordination**: Pipeline state management and messaging
- **Bytecode Compilation**: Optimized performance with compiled Python
- **Docker Orchestration**: Containerized deployment with automatic restarts
- **GPU Support**: NVIDIA GPU acceleration for inference
- **Evidence Upload (R3)**: Clip on detection, finalize after 5s no-detect, upload via multipart to NestJS (R2)
- **NVDEC/TURN**: Hardware decode (nvdec) and TURN stream support

## Quick Start

### 1. Build the System

```bash
# Build all Docker images
make build-all
```

### 2. Configure Streams

Copy and edit the environment file:

```bash
cd docker
cp .env.example .env
# Edit .env with your camera URLs
```

### 3. Start the System

```bash
make start
```

### 4. Monitor

```bash
# View all logs
make logs

# Check system status
make status

# Monitor pipeline health
make monitor
```

## Configuration

### Stream URLs

Edit `docker/.env`:

```env
# Primary cameras (always running)
STREAM_1_PRIMARY_URL=rtsp://192.168.1.101:8554/stream
STREAM_2_PRIMARY_URL=rtsp://192.168.1.102:8554/stream

# Backup cameras (activate on failure)
STREAM_1_BACKUP_URL=rtsp://192.168.1.201:8554/stream
STREAM_2_BACKUP_URL=rtsp://192.168.1.202:8554/stream

# Output streams (H.264)
STREAM_1_OUTPUT_URL=rtsp://localhost:8554/stream1
STREAM_2_OUTPUT_URL=rtsp://localhost:8554/stream2
```

### Pipeline Settings

Environment variables in `docker-compose.yml`:

- `HEALTH_CHECK_INTERVAL`: Health check frequency (default: 10 seconds)
- `STREAM_TIMEOUT`: Stream failure threshold (default: 30 seconds)
- `USE_GPU`: Enable GPU acceleration (default: true)

## How It Works

### 1. Primary Operation

- 2 ML pipelines process primary streams continuously
- Each pipeline:
  - Reads H.264 encoded video
  - Runs YOLOv8 detection
  - Draws bounding boxes
  - Outputs processed H.264 stream
  - Reports health to Redis

### 2. Health Monitoring

The orchestrator continuously monitors:
- Frame reception timestamps
- Pipeline health status
- Container state
- Redis heartbeats

### 3. Automatic Failover

When a primary stream fails:
1. Orchestrator detects health check failure
2. Publishes failure event to Redis
3. Starts corresponding backup pipeline
4. Backup pipeline connects to backup stream
5. Output continues with minimal interruption

### 4. Recovery

When primary stream recovers:
- Primary pipeline can be manually restarted
- Backup pipeline continues until primary is healthy
- Seamless transition back to primary

### 5. Evidence Capture & Upload
- On first detection, buffer from start of detection window (pre-buffer configurable)
- While detections continue, extend clip
- After 5 seconds with no detection (`EVIDENCE_NO_DETECT_TIMEOUT`), clip is finalized and uploaded via multipart to NestJS R3 (`/storage/evidence/multipart/*`) to R2
- Uses NVDEC for decode when enabled; supports TURN streams via env config

## Bytecode Compilation

The system includes optimized bytecode compilation for faster startup:

```bash
# Manual compilation
python scripts/compile_bytecode.py

# With source removal (production)
python scripts/compile_bytecode.py --remove-source

# With .pyc to .py renaming
python scripts/compile_bytecode.py --rename-pyc
```

Benefits:
- 30-40% faster startup time
- Reduced memory footprint
- Smaller container images

## Output Streams

Access processed streams:

- **RTSP**: `rtsp://localhost:8554/stream1`, `rtsp://localhost:8554/stream2`
- **WebRTC**: `http://localhost:8889` (browser-compatible)
- **HLS**: `http://localhost:8888/stream1`, `http://localhost:8888/stream2`

## Monitoring Commands

```bash
# View orchestrator logs
make logs-orchestrator

# View Redis logs
make logs-redis

# Check pipeline health in Redis
docker exec -it bantai-redis redis-cli
> KEYS pipeline:*
> HGETALL pipeline:primary-stream-1:health

# Check detections
> HGETALL pipeline:primary-stream-1:detections
```

## Troubleshooting

### Pipeline won't start

```bash
# Check Docker logs
docker logs bantai-orchestrator

# Check Redis connection
docker exec -it bantai-redis redis-cli PING
```

### Stream connection issues

```bash
# Test stream manually
ffmpeg -i rtsp://your-camera-url -frames:v 1 -f null -

# Check network connectivity
docker exec -it bantai-orchestrator ping camera1
```

### GPU not detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check docker-compose GPU settings
cd docker
docker-compose config
```

## Performance

Expected throughput:
- **With GPU**: 30-60 FPS per stream @ 1080p
- **CPU only**: 10-15 FPS per stream @ 720p

Memory usage:
- ML Pipeline: ~2-3GB per instance
- Orchestrator: ~100MB
- Redis: ~50MB

## Production Deployment

1. **Set resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

2. **Enable authentication** for Redis:
   ```yaml
   command: redis-server --requirepass your_password
   ```

3. **Configure logging**:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

4. **Use secrets** for credentials (Docker Swarm/Kubernetes)

## Cleanup

```bash
# Stop system
make stop

# Remove everything (including volumes)
make clean
```

## License

See main repository LICENSE file.
