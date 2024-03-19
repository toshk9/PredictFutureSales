FROM ubuntu:20.04

LABEL maintainer="ivan.vishniak@bk.ru"
ENV ADMIN="ivan"

RUN apt-get -y update
RUN apt-get -y install nginx

FROM python:3.10

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app