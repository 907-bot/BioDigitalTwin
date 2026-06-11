#!/bin/bash
###############################################################################
# Start the BioDigitalTwin API server for testing.
# Waits until /health responds, prints the PID for later shutdown.
###############################################################################

set -e
cd "$(dirname "$0")/../.."

PORT="${PORT:-8000}"
PY="backend/.venv/bin/python"
LOG="/tmp/biodt_server.log"
PID_FILE="/tmp/biodt_server.pid"

# Kill any prior instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping prior server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

# Also kill anything on the port
if command -v lsof >/dev/null; then
    PORT_PID=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "Killing process on port $PORT (PID $PORT_PID)..."
        kill -9 $PORT_PID 2>/dev/null || true
        sleep 1
    fi
fi

echo "Starting server on port $PORT..."
PYTHONPATH=backend nohup $PY -m uvicorn app_main:app \
    --host 0.0.0.0 --port $PORT \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo "Server PID: $SERVER_PID"

# Wait for /health
echo "Waiting for /health..."
for i in {1..30}; do
    if curl -s -f "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "✓ Server is up at http://localhost:$PORT"
        echo "  - Swagger UI:    http://localhost:$PORT/docs"
        echo "  - OpenAPI spec:  http://localhost:$PORT/openapi.json"
        echo "  - Logs:          $LOG"
        echo "  - Stop:          kill $SERVER_PID  (or: $0 --stop)"
        exit 0
    fi
    sleep 1
done

echo "✗ Server failed to start within 30s"
echo "  Last 20 log lines:"
tail -20 "$LOG"
exit 1
