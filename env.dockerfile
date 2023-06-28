FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# ==================================================================
# apt tools
# ------------------------------------------------------------------
RUN APT_INSTALL="apt install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' \
        /etc/apt/sources.list && \
    apt update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        zip \
        unzip \
        vim \
        nano \
        wget \
        curl \
        git \
        aria2 \
        apt-transport-https \
        openssh-client \
        openssh-server \
        libopencv-dev \
        libsnappy-dev \
        tzdata \
        iputils-ping \
        net-tools \
        htop \
        graphviz

# ==================================================================
# install miniconda
# ------------------------------------------------------------------
RUN curl -o ~/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/anaconda.sh && \
    ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# ==================================================================
# change sources and install python
# ------------------------------------------------------------------
RUN conda config --set show_channel_urls yes && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda install -y python=3.8 && \
    conda update --all

# ==================================================================
# install mxnet and utils
# ------------------------------------------------------------------
RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116 && \
    pip3 install matplotlib>=3.1.3 && \
    pip3 install torch>=1.1.0 && \
    pip3 install opencv-python==4.1.2.30 && \
    pip3 install numpy>=1.18.1

WORKDIR /root
