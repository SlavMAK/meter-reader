FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*
COPY reader/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Models from AI-on-the-edge-device (pinned commit), see LICENSE-AI-on-the-edge.md
ARG AOE=https://raw.githubusercontent.com/jomjol/AI-on-the-edge-device/328d2c7da9811f8fea5751864dbb1a792d894c56/sd-card/config
ADD --chmod=644 ${AOE}/dig-cont_0900_s3_q.tflite ${AOE}/dig-class11_1910_s2_q.tflite \
    ${AOE}/dig-class100-0182-s2_q.tflite ${AOE}/ana-cont_1500_s2_q.tflite models/

COPY LICENSE-AI-on-the-edge.md reader/ ./
ENV DATA_DIR=/data MODELS_DIR=/app/models PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
CMD ["python", "app.py"]
