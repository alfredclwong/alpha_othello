FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy your app and library code into the container
COPY ./ ./

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir git+https://github.com/alfredclwong/othello
