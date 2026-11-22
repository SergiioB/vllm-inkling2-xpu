# Adds this plugin to a vLLM XPU image that already contains Inkling 2 Mini
# (vLLM main at or after c32251b5d1). Build from the root of this repository:
#   docker build --build-arg BASE=vllm-xpu:main -t vllm-xpu:inkling2 .
ARG BASE=vllm/vllm-openai-xpu:nightly
FROM ${BASE}
COPY . /tmp/vllm-inkling2-xpu
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/vllm-inkling2-xpu \
    && rm -rf /tmp/vllm-inkling2-xpu
