FROM python:3.11
WORKDIR /server

COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir

RUN apt-get update && apt-get upgrade -y && apt-get install -y ffmpeg
RUN pip install --upgrade ffmpeg-python

COPY entrypoint.sh /server/entrypoint.sh
RUN chmod +x /server/entrypoint.sh
COPY cookies.txt /server
COPY youtube.py /server
CMD ["/server/entrypoint.sh"]