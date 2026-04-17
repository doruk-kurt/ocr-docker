FROM vllm/vllm-openai:nightly

# git is needed for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Install newer Transformers so GLM-OCR is recognized, plus worker/runtime deps.
# vLLM nightly currently imports pandas during CLI startup.
RUN pip uninstall -y transformers || true \
 && pip install -U git+https://github.com/huggingface/transformers.git \
 && pip install -U git+https://github.com/zai-org/glm-ocr.git \
 && pip install runpod requests pillow pandas

# Pre-download model weights into the image so cold starts don't hit HuggingFace
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"
ENV HF_HUB_OFFLINE=1

# Persist vLLM compile cache on network volume to speed up cold starts
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm-cache
ENV MAX_MODEL_LEN=16384
ENV GPU_MEMORY_UTILIZATION=0.95
ENV SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
ENV ENFORCE_EAGER=0
ENV MAX_IMAGE_SIDE=2000
ENV USE_GLMOCR_SDK=1

# Force glm-ocr SDK to use local vLLM instead of MaaS.
RUN mkdir -p /root/.config/glm-ocr \
 && printf "pipeline:\n  maas:\n    enabled: false\n  ocr_api:\n    api_host: localhost\n    api_port: 8080\n" > /root/.config/glm-ocr/config.yaml

COPY handler.py /handler.py

EXPOSE 8080

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
