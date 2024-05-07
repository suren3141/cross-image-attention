
# docker run --gpus all --rm -it -p 8888:8888 \
# -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
# -v E:\Dataset\apolloscape\:/mnt/dataset \
# -v E:\suren\TUM\practical\git\vehicular-3d-shape-reconstruction\BAAM\weights\:/mnt/weights \
# baam bash

ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

## To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget nano unzip \
    && apt-get clean

COPY . /workspace/cross-image-attention/
WORKDIR /workspace/cross-image-attention/

RUN python -m pip install -r environment/requirements.txt

# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git
# RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# # Create the environment:
# COPY env.yml .
# RUN conda env create -f env.yml


# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "baam", "/bin/bash", "-c"]

# # pytorch
# RUN conda install --quiet pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge 

# # Demonstrate the environment is activated:
# RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# # # pytorch3d
# RUN conda install --quiet -c fvcore -c iopath -c conda-forge fvcore iopath
# RUN conda install --quiet -c bottler nvidiacub
# RUN conda install --quiet pytorch3d -c pytorch3d
# # RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# # Open3D
# # RUN pip install open3d==0.14.1

# ## BAAM
# COPY BAAM /workspace/BAAM
# WORKDIR /workspace/BAAM

# # RUN git clone https://github.com/gywns6287/BAAM.git
# # RUN pip install -r /workspace/BAAM/requirements.txt

# #COPY weights/ /workspace/BAAM/

# # ENV PYTHONPATH "$PYTHONPATH:/workspace/"

# ## dataset
# WORKDIR /mnt/dataset/
# RUN ln -s /mnt/dataset/ /workspace/BAAM/data/apollo

# ## weights
# WORKDIR /mnt/weights/
# RUN ln -s /mnt/weights /workspace/BAAM/weights

# ENV APOLLO_PATH "/workspace/BAAM/data/apollo"