FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/var/data
COPY app.py ./
COPY studydesk ./studydesk
COPY static ./static
RUN mkdir -p /var/data
EXPOSE 8000
CMD ["python", "app.py"]
