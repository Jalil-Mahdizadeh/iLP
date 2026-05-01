FROM tensorflow/tensorflow:2.7.1-gpu

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONPATH=/app \
    CUDA_CACHE_PATH=/work/.cuda_cache \
    CUDA_CACHE_MAXSIZE=2147483648 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1 \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY iLP_run.py README.md ./
COPY iLP ./iLP
COPY MolAI ./MolAI
COPY examples ./examples
COPY tests ./tests

WORKDIR /work

ENTRYPOINT ["python", "/app/iLP_run.py"]
CMD ["--help"]
