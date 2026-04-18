# GLM-OCR Docker Image for RunPod Serverless

Docker image for running [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B parameter OCR model) on [RunPod Serverless](https://www.runpod.io/serverless-gpu) using vLLM.

Model weights are baked into the image at build time for fast cold starts.

The worker prefers the official `glmocr` self-hosted pipeline when it can. That means document parsing requests use the SDK's layout detection plus structured post-processing on top of the locally hosted GLM-OCR model instead of calling the raw model alone.

## What's included

- **Base image:** `vllm/vllm-openai:nightly`
- **Model:** [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (MIT License)
- **Transformers:** v5+ dev branch (required by GLM-OCR)
- **Serving:** vLLM on port 8080
- **Parsing pipeline:** official `glmocr[selfhosted]` SDK with local layout detection

## Deploy on RunPod Serverless

1. Create a new Serverless endpoint on RunPod.
2. Select **Build from GitHub repo** and point it to this repository.
3. No container start command is needed — the `CMD` in the Dockerfile handles it.
4. (Optional) Set `HF_TOKEN` as an environment variable in RunPod's UI for faster model downloads during builds.

## Usage

GLM-OCR supports two prompt types:

### Document parsing

Extract raw content from documents using these prompts:

| Task    | Prompt                 |
|---------|------------------------|
| Text    | `Text Recognition:`    |
| Formula | `Formula Recognition:` |
| Table   | `Table Recognition:`   |

### Information extraction

Extract structured data by providing a JSON schema as the prompt. Example:

```
Please output the information in the image in the following JSON format:
{
    "name": "",
    "date": "",
    "total": ""
}
```

### API example (RunPod Queue endpoint)

This worker is a **RunPod Serverless Queue** worker, so requests must be sent to
RunPod's `/run` or `/runsync` endpoint and wrapped in `input`.

If you send raw OpenAI payloads directly, the worker logs:
`Job has missing field(s): id or input.`

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "zai-org/GLM-OCR",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/9/99/ReceiptSwiss.jpg"}},
            {"type": "text", "text": "Text Recognition:"}
          ]
        }
      ]
    }
  }'
```

### SDK-style parsing request

You can also send a direct document parsing payload. This path maps most closely to the official self-hosted SDK:

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "images": [
        "https://upload.wikimedia.org/wikipedia/commons/9/99/ReceiptSwiss.jpg"
      ],
      "prompt": "Text Recognition:"
    }
  }'
```

Accepted SDK-first input shapes:

- `{"input": "<image_or_pdf_url_or_path>"}` for a single source
- `{"input": {"images": "<image_or_pdf_url_or_path>"}}` for a single source
- `{"input": {"images": ["page1", "page2"]}}` for a multi-page document
- `{"input": {"messages": [...]}}` for OpenAI-style image requests; the worker extracts image inputs and prompt text for the SDK when possible

## Build locally

```bash
docker build -t glm-ocr .
docker run --gpus all -p 8080:8080 glm-ocr
```

## License

This Dockerfile is provided as-is. GLM-OCR is released under the [MIT License](https://huggingface.co/zai-org/GLM-OCR). The vLLM base image has its own license terms.
