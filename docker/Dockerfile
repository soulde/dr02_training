# syntax=docker/dockerfile:1.7

ARG ISAACLAB_BASE_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-rc1
FROM ${ISAACLAB_BASE_IMAGE}

USER root

ARG ISAACLAB_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ARG UV_VERSION=0.9.25
ARG TENSORBOARD_VERSION=2.21.0
ARG VISER_VERSION=1.1.1

ENV ISAACLAB_PYTHON=${ISAACLAB_PYTHON} \
    HOME=/tmp \
    XDG_CACHE_HOME=/workspace/cache \
    PYTHONUNBUFFERED=1 \
    PATH=/workspace/isaaclab/_isaac_sim/kit/python/bin:${PATH}

RUN test -x "${ISAACLAB_PYTHON}" && \
    ("${ISAACLAB_PYTHON}" -m pip uninstall -y rsl-rl-lib || true)

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gosu && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_VERSION=${UV_VERSION} sh && \
    rm -rf /var/lib/apt/lists/* && \
    uv --version | grep -F "${UV_VERSION}"

RUN uv pip install --python "${ISAACLAB_PYTHON}" \
    "tensorboard==${TENSORBOARD_VERSION}" "viser==${VISER_VERSION}"

COPY docker/docker-entrypoint.sh /usr/local/bin/dr02-docker-entrypoint
RUN chmod 0755 /usr/local/bin/dr02-docker-entrypoint

WORKDIR /workspace/output
ENTRYPOINT ["/usr/local/bin/dr02-docker-entrypoint"]
CMD ["bash"]
