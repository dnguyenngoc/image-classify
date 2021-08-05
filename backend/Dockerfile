FROM python:3.9.5-buster

LABEL maintainer="Duy Nguyen <duyngunyenngoc@hotmail.com>"

COPY ./app /app

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install tesseract-ocr -y

COPY ./requirements.txt ./app/requirements.txt

COPY ./app/tesseract_vi/vie.traineddata /usr/share/tesseract-ocr/4.00/tessdata/vie.traineddata

WORKDIR /app

RUN pip3 install -r requirements.txt
    
EXPOSE 8082