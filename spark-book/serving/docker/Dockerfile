ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY serving/docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY sharing/core.py /app/sharing/core.py
COPY serving/infer.py serving/server.py /app/serving/
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=2s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/readyz',timeout=1).read()"]
ENTRYPOINT ["python", "/app/serving/server.py", "--host", "0.0.0.0", "--port", "8080"]
