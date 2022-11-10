ARG FROM_BASE_IMAGE=nvcr.io/nvidia/pytorch:22.05-py3
FROM ${FROM_BASE_IMAGE}
RUN pip install -U tritonclient[all] jupyterlab ipywidgets flair