# syntax=docker/dockerfile:1.7

ARG ISAACLAB_BASE_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-rc1
FROM ${ISAACLAB_BASE_IMAGE}

USER root

ARG ISAACLAB_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ARG UV_VERSION=0.9.25
ARG TENSORBOARD_VERSION=2.21.0
ARG VISER_VERSION=1.1.1
ARG ISAACSIM_PYTHON_PACKAGE_VERSION=6.1.0.0

ENV ISAACLAB_PYTHON=${ISAACLAB_PYTHON} \
    HOME=/tmp \
    XDG_CACHE_HOME=/workspace/cache \
    PYTHONUNBUFFERED=1 \
    DR02_STANDALONE_PXR_SCHEMA_PATH=/isaac-sim/kit/python/lib/python3.12/site-packages/isaacsim/asset/importer/utils/impl/schemas \
    DR02_MJC_PXR_PLUGIN_PATH=/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.12/site-packages/mujoco_usd_converter/plugins \
    PATH=/workspace/isaaclab/_isaac_sim/kit/python/bin:${PATH}

RUN test -x "${ISAACLAB_PYTHON}" && \
    ("${ISAACLAB_PYTHON}" -m pip uninstall -y rsl-rl-lib || true)

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gosu && \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh && \
    rm -rf /var/lib/apt/lists/* && \
    uv --version | grep -F "${UV_VERSION}"

RUN uv pip install --python "${ISAACLAB_PYTHON}" \
    "tensorboard==${TENSORBOARD_VERSION}" "viser==${VISER_VERSION}"

# Isaac Sim 6.1 ships extension schemas separately from the base Isaac Lab image.
RUN "${ISAACLAB_PYTHON}" -m pip install --extra-index-url https://pypi.nvidia.com \
        "isaacsim-robot-schema==${ISAACSIM_PYTHON_PACKAGE_VERSION}" \
        "mujoco-usd-converter==0.5.0" "urdf-usd-converter==0.3.2" \
        "usd-optimize==1.2.1" "tinyobjloader==2.0.0rc13" && \
    "${ISAACLAB_PYTHON}" -m pip install --no-deps --extra-index-url https://pypi.nvidia.com \
        "isaacsim-asset-isolated==${ISAACSIM_PYTHON_PACKAGE_VERSION}"

# The bundled regular isaacsim package masks this wheel's PEP 420 namespace portion.
RUN ln -s /isaac-sim/kit/python/lib/python3.12/site-packages/isaacsim/asset \
        /isaac-sim/python_packages/isaacsim/asset

COPY docker/docker-entrypoint.sh /usr/local/bin/dr02-docker-entrypoint
RUN chmod 0755 /usr/local/bin/dr02-docker-entrypoint

WORKDIR /workspace/output
ENTRYPOINT ["/usr/local/bin/dr02-docker-entrypoint"]
CMD ["bash"]
