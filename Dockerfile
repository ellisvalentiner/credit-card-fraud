FROM floydhub/dl-docker:cpu

MAINTAINER "Ellis Valentiner"

ADD . /root/project

RUN curl -o /root/project/creditcard.csv.zip "https://www.kaggle.com/dalpozz/creditcardfraud/downloads/creditcard.csv.zip"
RUN pip install --upgrade pip
RUN pip install -r /root/project/requirements.txt

WORKDIR /root/project
CMD ["ipython", "src/model.py"]
