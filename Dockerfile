FROM python:3.7-slim

RUN apt-get update -qq && apt-get install --no-install-recommends -y wget g++

ENV PROJECT_DIR /usr/src/app_glda

RUN mkdir -p ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

COPY requirements.txt ${PROJECT_DIR}

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . ${PROJECT_DIR}

ENTRYPOINT ["sh", ""]