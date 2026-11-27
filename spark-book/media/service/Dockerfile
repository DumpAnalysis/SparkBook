FROM nvcr.io/nvidia/pytorch@sha256:307c449df7dabed6b36e017a2b2f2e45bdec9dc203b81507111315f1e971de6e
USER root
COPY media/service/requirements.txt /opt/media-requirements.txt
RUN python -m pip uninstall -y flash-attn-4 quack-kernels && \
    python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/media-requirements.txt && \
    python -m pip check
COPY serving/docker/vllm_wsl_entrypoint.py /lab/bootstrap.py
COPY media/service/server.py /lab/service.py
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/tmp/hf PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 1000:1000
ENTRYPOINT ["python", "/lab/bootstrap.py"]
CMD ["python", "/lab/service.py"]
