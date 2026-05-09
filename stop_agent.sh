#!/usr/bin/env bash
# Stop the autonomous music intelligence agent
cd "$(dirname "$0")"

AGENT_PID=$(pgrep -f "agent.py" 2>/dev/null)

if [ -z "$AGENT_PID" ]; then
    echo "Agent not running."
    exit 0
fi

echo "Stopping agent (PID: $AGENT_PID)..."
kill "$AGENT_PID"
echo "Agent stopped."
