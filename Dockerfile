FROM gcr.io/kaggle-gpu-images/python:latest 

RUN pip install -U pip && \
    pip install jupyter-contrib-nbextensions && \
    jupyter contrib nbextension install && \
    jupyter nbextensions_configurator enable

COPY requirements.txt /tmp/
RUN pip install --quiet --no-cache-dir --requirement /tmp/requirements.txt

# make directory
RUN mkdir -p ~/.jupyter/lab/user-settings/jupyterlab_code_formatter
RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
