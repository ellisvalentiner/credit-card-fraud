FROM floydhub/fl-docker:cpu

MAINTAINER "Ellis Valentiner"

ADD . /root/project

RUN pip install --upgrade pip
RUN pip install -r /root/project/requirements.txt

CMD ["KERAS_BACKEND=tensorflow", "ipython", "/root/project/src/model.py"]
