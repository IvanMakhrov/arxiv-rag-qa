#!/bin/bash

# =============================================================================
# Kubernetes Port Forward Manager - RAG Project
# =============================================================================
# Usage:
#   ./port-forward.sh start   - Start all port forwards
#   ./port-forward.sh stop    - Stop all port forwards
#   ./port-forward.sh status  - Check running port forwards
# =============================================================================

NAMESPACE="rag-project"
PID_FILE="/tmp/k8s-port-forward-pids.txt"

# Service mappings: "Service Name:Service Resource:Local Port:Remote Port"
SERVICES=(
    "RAG UI:svc/rag-service:8000:8000"
    "Airflow:svc/airflow-webserver:8081:8080"
    "MLflow:svc/mlflow-service:5000:5000"
    "Qdrant:svc/qdrant-service:6333:6333"
    "Grafana:svc/grafana-service:3000:3000"
    "Prometheus:svc/prometheus:9090:9090"
    "MinIO:svc/minio-service:9001:9001"
    "Nginx:svc/nginx-service:8080:80"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Start all port forwards
start_port_forwards() {
    print_header "Starting Port Forwards"

    # Clear existing PID file
    : > "$PID_FILE"

    for service in "${SERVICES[@]}"; do
        IFS=':' read -r name resource local_port remote_port <<< "$service"

        echo -n "Starting $name (localhost:$local_port -> $resource:$remote_port)... "

        kubectl port-forward "$resource" "$local_port:$remote_port" -n "$NAMESPACE" > /dev/null 2>&1 &
        PID=$!
        echo $PID >> "$PID_FILE"

        # Wait briefly to check if port forward started successfully
        sleep 1
        if kill -0 $PID 2>/dev/null; then
            print_success "Running (PID: $PID)"
        else
            print_error "Failed to start"
        fi
    done

    echo ""
    print_header "Access URLs"
    echo -e "${GREEN}Nginx (RAG): http://localhost:8080${NC}"    # ← Added
    echo -e "${GREEN}RAG UI:      http://localhost:8000${NC}"
    echo -e "${GREEN}Airflow:     http://localhost:8081${NC}"
    echo -e "${GREEN}MLflow:      http://localhost:5000${NC}"
    echo -e "${GREEN}Qdrant:      http://localhost:6333${NC}"
    echo -e "${GREEN}Grafana:     http://localhost:3000${NC}"
    echo -e "${GREEN}Prometheus:  http://localhost:9090${NC}"
    echo -e "${GREEN}MinIO:       http://localhost:9001${NC}"
    echo ""
    print_info "To stop all port forwards, run: ./port-forward.sh stop"
    print_info "PIDs saved to: $PID_FILE"
}

# Stop all port forwards
stop_port_forwards() {
    print_header "Stopping Port Forwards"

    if [ ! -f "$PID_FILE" ]; then
        print_error "No PID file found. No port forwards to stop."
        exit 1
    fi

    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            print_success "Stopped PID: $pid"
        else
            print_info "PID $pid not running"
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    print_success "All port forwards stopped!"
}

# Check status of port forwards
check_status() {
    print_header "Port Forward Status"

    if [ ! -f "$PID_FILE" ]; then
        print_error "No PID file found. Port forwards may not be running."
        exit 1
    fi

    running=0
    stopped=0

    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            print_success "PID $pid is running"
            ((running++))
        else
            print_error "PID $pid is not running"
            ((stopped++))
        fi
    done < "$PID_FILE"

    echo ""
    print_info "Running: $running | Stopped: $stopped"
}

# Show help
show_help() {
    echo "Kubernetes Port Forward Manager"
    echo ""
    echo "Usage: $0 {start|stop|status|help}"
    echo ""
    echo "Commands:"
    echo "  start   - Start all port forwards"
    echo "  stop    - Stop all port forwards"
    echo "  status  - Check running port forwards"
    echo "  help    - Show this help message"
    echo ""
    echo "Services:"
    for service in "${SERVICES[@]}"; do
        IFS=':' read -r name resource local_port remote_port <<< "$service"
        echo "  $name: localhost:$local_port -> $resource:$remote_port"
    done
}

# Main script logic
case "${1:-}" in
    start)
        start_port_forwards
        ;;
    stop)
        stop_port_forwards
        ;;
    status)
        check_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Invalid command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
