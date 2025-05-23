FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Make a directory for storing AI code
RUN mkdir -p /app/ai

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip install --no-cache-dir git+https://github.com/alfredclwong/othello
