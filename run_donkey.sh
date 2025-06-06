#!/bin/bash

echo "Setting up Donkey-DSL Orchestrator..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directory if it doesn't exist
mkdir -p data

# Check if trajectories file exists
if [ ! -f "data/trajectories.json" ]; then
    echo "No trajectories file found. Using sample data..."
fi

# Start the server
echo "Starting Donkey-DSL API server on port 5000..."
python donkey_dsl.py