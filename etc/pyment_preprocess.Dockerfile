FROM docker.io/estenhl/pyment-preprocess:latest

USER root

RUN /envs/fastsurfer/bin/pip uninstall -y torch torchvision

RUN /envs/fastsurfer/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision

RUN /envs/fastsurfer/bin/pip install --no-cache-dir \
    --upgrade nvidia-cudnn-cu13