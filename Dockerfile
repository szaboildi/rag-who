# To build & run locally
### GAR_IMAGE=ragwho
### docker build --tag=$GAR_IMAGE:dev .
### docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev -> include sh on the end to enter shell to test ls and pip list

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# WORKDIR /prod

# Copy over all dependencies
COPY prompts prompts
COPY parameters_remote.toml parameters_remote.toml
COPY pyproject.toml pyproject.toml
COPY .env .env
COPY src src

RUN uv pip install . --system

CMD uvicorn ragwho.api.fast:api --host 0.0.0.0 --port $PORT
