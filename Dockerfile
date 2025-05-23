FROM python:3.13-slim

# Make a directory for storing AI code
RUN mkdir -p /app/ai

# Install git and g++
RUN apt-get update && apt-get install -y git g++ make && rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip install --no-cache-dir git+https://github.com/alfredclwong/othello

COPY Egaroucid4 /app/Egaroucid4
WORKDIR /app/Egaroucid4/src
RUN g++ egaroucid4.cpp -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -lpthread -Wall -o egaroucid4.out
