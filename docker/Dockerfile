# Classical tracker: CPU only, OpenCV + NumPy.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg > /dev/null && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install --quiet -r /tmp/requirements.txt -r /tmp/requirements-dev.txt ruff

WORKDIR /app
