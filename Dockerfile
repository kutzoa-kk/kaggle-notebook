FROM gcr.io/kaggle-gpu-images/python:latest 

RUN pip install -U pip && \
    pip install black && \
    pip install jupyter-contrib-nbextensions && \
    jupyter contrib nbextension install && \
    jupyter nbextensions_configurator enable && \
    jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip && \
    jupyter nbextension enable jupyter-black-master/jupyter-black

COPY requirements.txt /tmp/
RUN pip install --quiet --no-cache-dir --requirement /tmp/requirements.txt