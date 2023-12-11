#!/bin/bash

while true; do
    # Read content from last_old_time.txt
    content=$(cat last_old_time.txt)

    # Append content to usage.txt along with a timestamp
    echo "$(date): $content" >> usage.txt

    # Sleep for 10 minutes
    sleep 600
done
