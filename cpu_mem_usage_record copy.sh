#!/bin/bash

# Set the name of the process you want to monitor
process_name="python"

# Create a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# Create a file to store the usage data
output_file="process_usage.log"

# Loop infinitely to monitor the process
while true; do
  # Get the process ID (PID) of the "python" process
  pid=878083

  if [ -n "$pid" ]; then
    # Get the current timestamp
    current_time=$(timestamp)
    
    # Get CPU and memory usage using top
    cpu_usage="$(top -p $pid -b -n 1 | grep $pid | awk '{print $9}')"
    memory_usage="$(top -p $pid -b -n 1 | grep $pid | awk '{print $10}')"
    # Append the data to the output file
    echo "$current_time - CPU Usage: $cpu_usage%, Memory Usage: $memory_usage%" >> "$output_file"
  else
    # If the process is not found, log it
    echo "$(timestamp) - $process_name is not running." >> "$output_file"
  fi

  # Sleep for 3 minutes
  sleep 180
done
