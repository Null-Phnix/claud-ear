#!/bin/bash
# Stop the Music Intelligence Agent
if pgrep -f "python agent.py" > /dev/null; then
    pkill -f "python agent.py"
    echo "Agent stopped."
else
    echo "Agent is not running."
fi
