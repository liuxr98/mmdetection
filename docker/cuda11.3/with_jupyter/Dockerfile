FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Install basic softwares
RUN apt-get update && apt-get install -y vim ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 build-essential wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install flask and notebook
RUN pip install --no-cache-dir flask jupyter jupyter_contrib_nbextensions\
    && jupyter contrib nbextension install --user \
    && jupyter notebook --generate-config \
    && echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip='*'" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password=''" >> ~/.jupyter/jupyter_notebook_config.py \
    && conda clean -a -y
    
RUN mkdir /workdir
WORKDIR /workdir

CMD ["jupyter","notebook","--no-browser","--allow-root","--NotebookApp.token=''"]