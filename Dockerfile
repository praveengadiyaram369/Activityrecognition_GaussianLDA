FROM python:3.7-slim

RUN apt-get update -qq && apt-get install --no-install-recommends -y wget g++ && apt-get install -y git

ENV PROJECT_DIR /usr/src/app_glda

RUN mkdir -p ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

COPY requirements.txt ${PROJECT_DIR}

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install git+git://github.com/jcrudy/choldate.git#egg=choldate
RUN pip install git+https://github.com/praveengadiyaram369/gaussianlda.git@topic_dist_per_doc#egg=gaussianlda


COPY API ${PROJECT_DIR}
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]