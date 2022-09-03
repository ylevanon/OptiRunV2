FROM python:3.7-alpine

ADD main.py .

RUN pip install pandas

CMD ["python", "./main.py"]