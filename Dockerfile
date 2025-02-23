FROM selenium/standalone-chrome:latest
# FROM python:3.12.9-slim-bullseye

RUN sudo apt update && sudo apt -y upgrade
RUN sudo apt -y install git
# RUN apt -y install chromium

# WORKDIR /home/
# COPY . /home/
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
