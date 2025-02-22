# FROM selenium/standalone-chrome:latest
FROM python:3.12.9-slim-bullseye

RUN apt update && apt -y upgrade
RUN apt -y install git
RUN apt -y install chromium

WORKDIR /home/
COPY . /home/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
