FROM jupyter/datascience-notebook

COPY requirements.txt /tmp/
RUN pip install -U pip && \
    pip install --quiet --no-cache-dir --requirement /tmp/requirements.txt

RUN jupyter lab build
# make directory
RUN mkdir -p ~/.jupyter/lab/user-settings/jupyterlab_code_formatter
RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
