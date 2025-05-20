FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy your app and library code into the container
COPY ./ ./

# Install pip packages
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir tqdm
RUN pip install --no-cache-dir .
