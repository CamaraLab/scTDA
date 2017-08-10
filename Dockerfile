FROM jupyter/scipy-notebook
VOLUME /home/jovyan/work
ADD . /scTDA
USER root
RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*
WORKDIR /scTDA
RUN $CONDA_DIR/envs/python2/bin/python /scTDA/setup.py install
USER $NB_USER
WORKDIR /home/jovyan/work
