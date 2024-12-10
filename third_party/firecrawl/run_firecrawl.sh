#!/bin/bash

# Start the Redis server
redis-server &

# Initialize workers
pnpm run workers &

# Run the main server
pnpm run start &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?