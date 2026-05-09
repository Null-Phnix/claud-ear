#!/usr/bin/env bash
# Pause agent during peak hours (1:30 PM) to avoid GPU congestion
# Used by the music intelligence agent for power scheduling

# Check if agent process is running
AGENT_PID=$(pgrep -f "agent.py" 2>/dev/null)

if [ -z "$AGENT_PID" ]; then
    echo "Agent not running. Nothing to pause."
    exit 0
fi

echo "Pausing agent (PID $AGENT_PID) at $(date)"
kill -STOP "$AGENT_PID"
echo "Agent paused. Run stop_agent.sh to resume/stop."
