FROM nvidia/cuda:11.7.1-devel-ubuntu18.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install pytorch
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install basic softwares
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install mmcv
RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN pip install -r requirements/optional.txt
RUN MMCV_WITH_OPS=1 pip install -e . -v

# Install MMDetection


