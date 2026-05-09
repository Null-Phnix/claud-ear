#!/bin/bash
# Start the Music Intelligence Agent in the background
cd /Users/josii/Desktop/audio-mcp-review

# Check if already running
if pgrep -f "python agent.py" > /dev/null; then
    echo "Agent is already running (PID $(pgrep -f 'python agent.py'))"
    exit 1
fi

nohup env -u CLAUDECODE uv run python agent.py >> agent.log 2>&1 &
PID=$!
echo "Agent started (PID $PID)"
echo "Logs: tail -f /Users/josii/Desktop/audio-mcp-review/agent.log"
