# Use the latest PyTorch base image with CUDA 12.1 and cuDNN 8
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libpython3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy files into workspace
COPY . /workspace

# Set the working directory
WORKDIR /workspace
