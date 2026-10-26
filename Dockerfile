# The whole pipeline in a container:
#
#   docker build -t pdeeg .
#   docker run --rm -v "$PWD/data:/app/data" pdeeg            # pdeeg all
#   docker run --rm -v "$PWD/data:/app/data" pdeeg stats      # or any other command
#
# Everything the pipeline writes goes to the mounted data/ directory: the dataset and what is
# computed from it as usual, and, because /app/reports, /app/mlruns and /app/mlflow.db link
# into it, the reports (data/reports) and the MLflow runs (data/mlruns, data/mlflow.db).
# On Linux, add --user "$(id -u):$(id -g)" when the host's data/ is not writable by uid 1000.

# Python 3.12, as in development; the digest pins the exact image.
FROM python:3.12.14-slim-trixie@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    MNE_DONTWRITE_HOME=true \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# The pinned dependencies first, in their own layer, so that a change to the code does not
# install them again.
COPY pyproject.toml README.md ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY docker/entrypoint.sh ./docker/entrypoint.sh

# The code stays owned by root; the user may write the data volume and, for SQLite's journal
# next to the linked database, /app itself.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin pdeeg \
    && mkdir data \
    && ln -s data/reports reports \
    && ln -s data/mlruns mlruns \
    && ln -s data/mlflow.db mlflow.db \
    && ln -s data/models models \
    && ln -s data/predictions predictions \
    && chown pdeeg:pdeeg /app /app/data

USER pdeeg
VOLUME ["/app/data"]
ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
CMD ["all"]
