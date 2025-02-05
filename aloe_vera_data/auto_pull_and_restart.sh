#!/bin/bash

# Navigate to the project directory
cd /home/tomo/Documents/piprojects

# Pull the latest changes from GitHub
git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

# If there are changes, pull and restart the service
if [ $LOCAL != $REMOTE ]; then
    echo "Changes detected, pulling updates..."
    git pull origin main
    sudo systemctl restart myscript.service
fi
