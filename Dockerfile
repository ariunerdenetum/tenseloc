# Base image with PyTorch + CUDA for A100/V100
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set CUDA path explicitly
ENV CUDA_HOME=/usr/local/cuda
# Ensure UTF-8 locale
ENV LANG C.UTF-8

# Install system dependencies and cleanup
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      htop \
      gnupg \
      curl \
      ca-certificates \
      vim \
      tmux && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python libs
# RUN python3 -m pip install --upgrade pip && \
#     python3 -m pip install \
#       transformers \
#       pandas \
#       numpy \
#       accelerate \
#       autopep8
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt

# Create non-root user matching host UID/GID
ARG USER_UID
ARG USER_NAME
ENV USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USER_NAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home ${USER_NAME} && \
    mkdir -p /home/${USER_NAME}/.local && chown -R ${USER_UID}:${USER_GID} /home/${USER_NAME}

# Switch to non-root user
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# Default shell
CMD ["/bin/bash"]
