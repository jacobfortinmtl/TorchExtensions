FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN git clone --recursive https://github.com/jacobfortinmtl/pytorch.git \
&& cd pytorch \
&& git submodule sync \
&& git submodule update --init --recursive

# Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    /bin/bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh
    
# Initialize Conda
COPY conda_env.yml conda_env.yml
RUN /opt/conda/bin/conda env create -f conda_env.yml
# Ensure the conda environment is activated for subsequent RUN commands
SHELL ["conda", "run", "-n", "torch_source", "/bin/bash", "-c"]

# Copy PyTorch requirements.txt and install PyTorch dependencies
RUN conda run -n torch_source pip install -r /pytorch/requirements.txt

RUN conda run -n torch_source conda config --add channels conda-forge && \
    conda run -n torch_source conda install -y \
    mkl-static \
    mkl-include \
    && conda run -n torch_source conda clean -ya

# Fix ownership for git safety
RUN git config --global --add safe.directory '*'


ARG MAX_JOBS=$(nproc)
ARG USE_CUDA=0

# Additional instructions
RUN /bin/bash -c "source activate torch_source && \
                  cd /pytorch && \
                  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
                  DEBUG=1 MAX_JOBS=${MAX_JOBS} USE_CUDA=${USE_CUDA} python setup.py develop --verbose"