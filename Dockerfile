FROM floydhub/dl-docker:cpu

MAINTAINER "Ellis Valentiner"

ADD . /root/project

RUN pip install --upgrade pip
RUN pip install -r /root/project/requirements.txt

WORKDIR /root/project
CMD ["ipython", "src/model.py"]
