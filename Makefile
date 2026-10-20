.PHONY: build-pipeline build-orchestrator build-all start stop restart logs clean

# Build ML pipeline image
build-pipeline:
	@echo "Building ML pipeline image..."
	docker build -t bantai-ml-pipeline:latest -f docker/Dockerfile.ml-pipeline .

# Build orchestrator image
build-orchestrator:
	@echo "Building orchestrator image..."
	docker build -t bantai-orchestrator:latest -f docker/Dockerfile.orchestrator .

# Build all images
build-all: build-pipeline build-orchestrator
	@echo "All images built successfully"

# Start the pipeline system
start:
	@echo "Starting BANTAI ML pipeline system..."
	cd docker && docker-compose up -d
	@echo "System started. Use 'make logs' to view logs"

# Stop the pipeline system
stop:
	@echo "Stopping pipeline system..."
	cd docker && docker-compose down

# Restart the system
restart: stop start

# View logs
logs:
	cd docker && docker-compose logs -f

# View specific service logs
logs-orchestrator:
	cd docker && docker-compose logs -f orchestrator

logs-redis:
	cd docker && docker-compose logs -f redis

# Clean up everything
clean:
	@echo "Cleaning up containers, images, and volumes..."
	cd docker && docker-compose down -v
	docker rmi bantai-ml-pipeline:latest bantai-orchestrator:latest || true
	@echo "Cleanup complete"

# Monitor pipeline health
monitor:
	@echo "Monitoring pipeline health..."
	docker exec -it bantai-redis redis-cli --scan --pattern "pipeline:*"

# Check system status
status:
	@echo "=== Pipeline System Status ==="
	@docker ps --filter "name=bantai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
