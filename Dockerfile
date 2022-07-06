FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    apt-utils \
    gcc \
    g++ \
    make \
    cmake \
    vim vim-gtk3 \
    git \
    wget \
    curl \
    xvfb \
    lsb-release \
    sudo \
    ffmpeg \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf


RUN useradd --create-home dockeruser \
    && echo "dockeruser:docker" | chpasswd \
    && adduser dockeruser sudo
USER dockeruser
WORKDIR /home/dockeruser


ARG CONDA=miniconda.sh
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O $CONDA \
    && bash $CONDA -b -p ~/opt/miniconda \
    && rm $CONDA


RUN mkdir -p /home/dockeruser/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /home/dockeruser/.mujoco \
    && rm mujoco.tar.gz \
    && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dockeruser/.mujoco/mujoco210/bin:/usr/lib/nvidia' \
        >> ~/.bashrc


COPY --chown=dockeruser:dockeruser . /home/dockeruser/metaworld_door_env
WORKDIR /home/dockeruser/metaworld_door_env


RUN eval "$(~/opt/miniconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda create -n dooropen python=3.9 \
    && conda activate dooropen \
    && pip install -r requirements.txt \
    && echo "conda activate dooropen" >> ~/.bashrc


ENTRYPOINT ["/bin/bash"]
