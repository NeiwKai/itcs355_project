FROM python@sha256:a2bc8c35469b6fe37735f7c4dae39049470b2ce068e73f799c02452de31d24c6 AS builder


ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 

WORKDIR /build

COPY requirements.txt ./

RUN pip install --prefix=/install --require-hashes -r requirements.txt


FROM python@sha256:a2bc8c35469b6fe37735f7c4dae39049470b2ce068e73f799c02452de31d24c6 AS runtime

RUN useradd --create-home --uid 10001 runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
WORKDIR /app
RUN mkdir -p /app/mlruns && chown -R runner:runner /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner scripts/ ./scripts/

USER runner

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--C", "1.0", "--epsilon", "0.1"]
