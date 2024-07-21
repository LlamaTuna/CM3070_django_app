#!/bin/bash

# Check if the port 8000 is in use
PID=$(sudo lsof -t -i :8000)

# If a process is found, kill it
if [ ! -z "$PID" ]; then
  echo "Killing process with PID: $PID"
  sudo kill -9 $PID
  echo "Process killed."
else
  echo "No process found using port 8000."
fi
