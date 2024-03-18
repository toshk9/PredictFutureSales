FROM jupyter/datascience-notebook

WORKDIR /

USER root

RUN pip install catboost gdown

COPY . /home/jovyan/work

USER jovyan

EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.notebook_dir='/home/jovyan/work'"]