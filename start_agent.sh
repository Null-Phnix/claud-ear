#!/usr/bin/env bash
# Start the autonomous music intelligence agent
set -e

cd "$(dirname "$0")"

echo "Starting Music Intelligence Agent..."
echo "Music dir: $HOME/Documents/music/music data/"
echo "Analyses dir: $HOME/Documents/music/analyses/"

# Run the agent in background
nohup python agent.py >> agent.log 2>&1 &

AGENT_PID=$!
echo "Agent started (PID: $AGENT_PID)"
echo "Logs: agent.log"
echo "Stop with: ./stop_agent.sh"
