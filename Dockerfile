FROM python:3.7

COPY * /

RUN pip3 install cv2, numpy

CMD [ "python3", "-u","./main.py" ]