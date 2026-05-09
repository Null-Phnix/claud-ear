#!/bin/bash
OUTPUT="/private/tmp/claude-501/-Users-josii-Desktop-audio-mcp-review/tasks/bdltsnmdi.output"
TARGET=130

echo "Watching for track $TARGET to complete..."
while true; do
    COUNT=$(grep -c "Done in" "$OUTPUT" 2>/dev/null || echo 0)
    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "Reached $TARGET tracks! Stopping transcription..."
        pkill -f "transcribe_all_remaining" && echo "Process stopped." || echo "Process already stopped."
        break
    fi
    sleep 30
done
