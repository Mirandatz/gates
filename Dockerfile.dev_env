FROM python:3.10.10-slim-buster as base

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install system deps
# curl to download stuff
# git
# graphviz to plot models
# libgl1 for opencv2
FROM base as with_system_deps
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    curl git graphviz libgl1 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create user and configure pyenv
FROM with_system_deps as with_user
ARG UNAME
ARG UID
ARG GID
RUN groupadd --gid $GID $UNAME
RUN useradd --create-home --uid $UID --gid $GID --shell /bin/bash $UNAME

# install python deps
FROM with_user as with_python_deps
COPY ./requirements /app/python_requirements
RUN python3 -m pip install -r /app/python_requirements/dev.txt --no-cache-dir
