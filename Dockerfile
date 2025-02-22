FROM selenium/standalone-chrome:latest
RUN sudo apt update && sudo apt -y upgrade && sudo apt install git
RUN pip install -r requirements.txt
