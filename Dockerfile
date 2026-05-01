FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CUDA_CACHE_PATH=/work/.cuda_cache \
    CUDA_CACHE_MAXSIZE=2147483648 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1 \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY iLP_run.py README.md CITATION.cff ./
COPY models/ilp ./models/ilp
COPY examples ./examples
COPY tests ./tests

WORKDIR /work

ENTRYPOINT ["python", "-m", "ilp.pipeline"]
CMD ["--help"]
