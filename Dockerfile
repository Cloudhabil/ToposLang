FROM python:3.13-slim

WORKDIR /app

COPY . /app

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PATH="/app/bin:${PATH}"

# Verify test suite on build
RUN python3 -m unittest discover tests

CMD ["tau-run", "examples/06_spatial_agents.tau"]
