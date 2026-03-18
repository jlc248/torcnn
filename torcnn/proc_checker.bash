#!/bin/bash

# --- CONFIGURATION ---
# The email address that will receive alerts
EMAIL="john.cintineo@noaa.gov"

# List the processes you want to monitor (exact names from 'ps' or 'pgrep')
SERVICES=("MAIN-TORPLIST" "MAIN-TORCNN")

# Get the hostname for the email subject
SERVER_NAME=$(hostname)
# ---------------------

OFFLINE_SERVICES=()

for SERVICE in "${SERVICES[@]}"; do
    # pgrep -x looks for an exact match of the process name
    if ! pgrep -f "$SERVICE" > /dev/null; then
        OFFLINE_SERVICES+=("$SERVICE")
    fi
done

# If the array of offline services is NOT empty, send an email
if [ ${#OFFLINE_SERVICES[@]} -gt 0 ]; then
    MESSAGE="Alert: The following processes are not running on $SERVER_NAME: ${OFFLINE_SERVICES[*]}"
    
    echo "$MESSAGE" | mail -s "CRITICAL: Process Down on $SERVER_NAME" "$EMAIL"
fi
