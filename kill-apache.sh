#!/bin/bash

# Find all Apache processes listening on port 80
pids=$(sudo lsof -t -i :80 -sTCP:LISTEN)

# Check if any processes are found
if [ -z "$pids" ]; then
  echo "No Apache processes found on port 80."
else
  echo "Killing Apache processes on port 80 with PIDs: $pids"

  # Kill each process found
  for pid in $pids; do
    sudo kill -9 $pid
    echo "Killed process with PID: $pid"
  done
fi
