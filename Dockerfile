ARG FROM_BASE_IMAGE=nvcr.io/nvidia/pytorch:22.05-py3
FROM ${FROM_BASE_IMAGE}
RUN pip install -U tritonclient[all]==2.27.0 jupyterlab==3.5.0 ipywidgets==8.0.2 flair==0.11.3