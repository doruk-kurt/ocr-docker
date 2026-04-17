"""
RunPod Serverless handler for GLM-OCR via vLLM.

Starts vLLM HTTP server in the background, waits for it to be ready,
then forwards incoming RunPod jobs to the local OpenAI-compatible API.
"""

import os
import sys
import time
import base64
import tempfile
import subprocess
import threading
import logging
from io import BytesIO
from urllib.parse import urlparse
import requests
import runpod
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

VLLM_PORT = 8080
VLLM_URL = f"http://localhost:{VLLM_PORT}"
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "16384")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
SPECULATIVE_CONFIG = os.getenv(
    "SPECULATIVE_CONFIG",
    '{"method": "mtp", "num_speculative_tokens": 1}',
)
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "0").lower() in {"1", "true", "yes"}
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2000"))
USE_GLMOCR_SDK = os.getenv("USE_GLMOCR_SDK", "1").lower() in {"1", "true", "yes"}

OCR_PARSER = None
VLLM_PROCESS = None
VLLM_STARTUP_ERROR = None
VLLM_READY = False


def _truncate_text(value, limit=1000):
    """Return a compact single-line string for logs and error payloads."""
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _collect_gpu_diagnostics():
    """Collect lightweight GPU visibility diagnostics for startup failures."""
    diagnostics = {
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "<unset>"),
        "NVIDIA_VISIBLE_DEVICES": os.getenv("NVIDIA_VISIBLE_DEVICES", "<unset>"),
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        diagnostics["nvidia_smi_exit_code"] = result.returncode
        if result.stdout.strip():
            diagnostics["nvidia_smi_stdout"] = _truncate_text(result.stdout, 500)
        if result.stderr.strip():
            diagnostics["nvidia_smi_stderr"] = _truncate_text(result.stderr, 500)
    except Exception as exc:
        diagnostics["nvidia_smi_error"] = str(exc)

    return diagnostics


def _format_gpu_diagnostics():
    diagnostics = _collect_gpu_diagnostics()
    return ", ".join(f"{key}={value}" for key, value in diagnostics.items())


def _set_vllm_startup_error(message):
    """Record a persistent startup error for later diagnostics."""
    global VLLM_STARTUP_ERROR, VLLM_READY
    VLLM_STARTUP_ERROR = message
    VLLM_READY = False


def _vllm_error_payload(job_id, message, **extra):
    """Return a structured RunPod response instead of surfacing a generic 500."""
    payload = {
        "error": message,
        "job_id": job_id,
        "vllm_url": f"{VLLM_URL}/v1/chat/completions",
    }

    if VLLM_STARTUP_ERROR:
        payload["startup_error"] = VLLM_STARTUP_ERROR
    if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is not None:
        payload["vllm_exit_code"] = VLLM_PROCESS.returncode

    payload.update({k: v for k, v in extra.items() if v not in {None, ""}})
    return payload


def stream_output(pipe):
    """Stream vLLM logs into worker logs for easier debugging."""
    try:
        for line in pipe:
            line = line.strip()
            if line:
                log.info("[vllm] %s", line)
    except Exception as exc:
        log.exception("Error while streaming vLLM logs: %s", exc)
    finally:
        pipe.close()


def start_vllm():
    """Start vLLM as a background process with log forwarding."""
    log.info("GPU diagnostics before vLLM start: %s", _format_gpu_diagnostics())
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--allowed-local-media-path", "/",
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--speculative-config", SPECULATIVE_CONFIG,
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")

    log.info("Starting vLLM: %s", " ".join(cmd))
    log.info(
        "vLLM context window configured to %s tokens (gpu_memory_utilization=%s)",
        MAX_MODEL_LEN,
        GPU_MEMORY_UTILIZATION,
    )
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        t = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        t.start()

    return process


def wait_for_vllm(process, timeout=600):
    """Wait for vLLM HTTP server and engine core to be ready."""
    global VLLM_READY
    start = time.time()
    last_error = ""
    while time.time() - start < timeout:
        if process.poll() is not None:
            message = (
                f"vLLM exited during startup with code {process.returncode}. "
                f"GPU diagnostics: {_format_gpu_diagnostics()}"
            )
            _set_vllm_startup_error(message)
            raise RuntimeError(message)
        try:
            health = requests.get(f"{VLLM_URL}/health", timeout=2)
            if health.status_code == 200:
                models = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
                if models.status_code == 200:
                    VLLM_READY = True
                    log.info("vLLM is ready")
                    return True
                last_error = (
                    f"/v1/models returned {models.status_code}: "
                    f"{_truncate_text(models.text)}"
                )
            else:
                last_error = (
                    f"/health returned {health.status_code}: "
                    f"{_truncate_text(health.text)}"
                )
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)

    message = f"vLLM did not become ready within {timeout}s"
    if last_error:
        message = f"{message}. Last readiness check: {last_error}"
    _set_vllm_startup_error(message)
    raise TimeoutError(message)


def init_glmocr_sdk():
    """Initialize glm-ocr SDK parser if available."""
    if not USE_GLMOCR_SDK:
        log.info("GLM-OCR SDK disabled by USE_GLMOCR_SDK")
        return None

    try:
        from glmocr import GlmOcr
    except Exception as exc:
        log.warning("Failed to import glm-ocr SDK: %s", exc)
        return None

    try:
        parser = GlmOcr()
        log.info("GLM-OCR SDK initialized")
        return parser
    except Exception as exc:
        log.warning("Failed to initialize glm-ocr SDK: %s", exc)
        return None


def _extract_image_url(content_part):
    """Return image URL string from an OpenAI content part."""
    if not isinstance(content_part, dict):
        return None
    if content_part.get("type") != "image_url":
        return None

    image_url = content_part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return image_url.get("url")
    return None


def _set_image_url(content_part, new_url):
    """Update image_url field while preserving OpenAI-compatible shape."""
    image_url = content_part.get("image_url")
    if isinstance(image_url, dict):
        image_url["url"] = new_url
    else:
        content_part["image_url"] = {"url": new_url}


def _extract_job_image_and_prompt(job_input):
    """Extract first image URL/path and prompt text from supported payload shapes."""
    image_ref = None
    prompt_parts = []

    if isinstance(job_input, str):
        return job_input, ""

    if not isinstance(job_input, dict):
        return None, ""

    image_ref = job_input.get("url") or job_input.get("image")
    prompt = job_input.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        prompt_parts.append(prompt.strip())

    messages = job_input.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                if content.strip():
                    prompt_parts.append(content.strip())
                continue
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue
                if image_ref is None:
                    image_ref = _extract_image_url(part)
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        prompt_parts.append(text.strip())

    return image_ref, "\n".join(prompt_parts).strip()


def _read_image_bytes(url):
    """Read image bytes from http(s), file://, or absolute local path."""
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content

    if parsed.scheme == "file":
        with open(parsed.path, "rb") as f:
            return f.read()

    if parsed.scheme == "" and url.startswith("/"):
        with open(url, "rb") as f:
            return f.read()

    raise ValueError(f"Unsupported image URL scheme: {parsed.scheme or 'relative-path'}")


def _resize_image_to_data_url(image_bytes, max_side):
    """Resize image if needed and return a data URL (or None if unchanged)."""
    with Image.open(BytesIO(image_bytes)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_side:
            return None, (width, height), (width, height)

        ratio = max_side / float(longest)
        new_size = (
            max(1, int(width * ratio)),
            max(1, int(height * ratio)),
        )
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        out = BytesIO()

        has_alpha = "A" in resized.getbands()
        if has_alpha:
            resized.save(out, format="PNG", optimize=True)
            mime = "image/png"
        else:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            resized.save(out, format="JPEG", quality=90, optimize=True)
            mime = "image/jpeg"

        encoded = base64.b64encode(out.getvalue()).decode("ascii")
        return f"data:{mime};base64,{encoded}", (width, height), new_size


def _resize_image_to_file_path(image_bytes, max_side):
    """
    Resize image to MAX_IMAGE_SIDE and store in a temporary local file.
    Returns (path_or_none, old_size, new_size).
    """
    with Image.open(BytesIO(image_bytes)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_side:
            return None, (width, height), (width, height)

        ratio = max_side / float(longest)
        new_size = (
            max(1, int(width * ratio)),
            max(1, int(height * ratio)),
        )
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        has_alpha = "A" in resized.getbands()
        if has_alpha:
            suffix = ".png"
            save_kwargs = {"format": "PNG", "optimize": True}
        else:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            suffix = ".jpg"
            save_kwargs = {"format": "JPEG", "quality": 90, "optimize": True}

        tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="glmocr_",
            delete=False,
        )
        with tmp:
            resized.save(tmp, **save_kwargs)
        return tmp.name, (width, height), new_size


def _prepare_image_for_sdk(image_ref, job_id):
    """Return image path/url for SDK parse and list of temp files to clean up."""
    cleanup_paths = []
    if MAX_IMAGE_SIDE <= 0:
        return image_ref, cleanup_paths

    try:
        image_bytes = _read_image_bytes(image_ref)
        resized_path, old_size, new_size = _resize_image_to_file_path(
            image_bytes, MAX_IMAGE_SIDE
        )
        if resized_path is None:
            return image_ref, cleanup_paths

        cleanup_paths.append(resized_path)
        log.info(
            "Job %s: SDK image resized from %sx%s to %sx%s",
            job_id,
            old_size[0],
            old_size[1],
            new_size[0],
            new_size[1],
        )
        return resized_path, cleanup_paths
    except Exception as exc:
        log.warning("Job %s: SDK image resize skipped (%s)", job_id, exc)
        return image_ref, cleanup_paths


def _normalize_sdk_result(result):
    """Normalize glm-ocr SDK output into stable response keys."""
    if isinstance(result, dict):
        layout_json = result.get("json_result") or result.get("layout_json")
        markdown = result.get("md_result") or result.get("markdown")
        return layout_json, markdown, result

    layout_json = getattr(result, "json_result", None)
    markdown = getattr(result, "md_result", None)
    raw = {
        "json_result": layout_json,
        "md_result": markdown,
    }
    return layout_json, markdown, raw


def _parse_with_sdk(job_input, job_id):
    """Parse image with glm-ocr SDK and return structured output dict or None."""
    if OCR_PARSER is None:
        return None

    image_ref, prompt = _extract_job_image_and_prompt(job_input)
    if not image_ref:
        return None

    image_input, cleanup_paths = _prepare_image_for_sdk(image_ref, job_id)
    try:
        # Some SDK versions support prompt kwarg; fall back to image-only parse.
        if prompt:
            try:
                result = OCR_PARSER.parse(image_input, prompt=prompt)
            except TypeError:
                result = OCR_PARSER.parse(image_input)
        else:
            result = OCR_PARSER.parse(image_input)

        layout_json, markdown, raw = _normalize_sdk_result(result)
        pages = len(layout_json) if isinstance(layout_json, list) else 1
        return {
            "layout_json": layout_json,
            "markdown": markdown,
            "pages": pages,
            "raw": raw,
        }
    finally:
        for path in cleanup_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def preprocess_images(job_input, job_id):
    """
    Resize image_url content parts to reduce visual token usage.
    Disabled when MAX_IMAGE_SIDE <= 0.
    """
    if MAX_IMAGE_SIDE <= 0:
        return

    messages = job_input.get("messages")
    if not isinstance(messages, list):
        return

    seen = 0
    resized = 0
    skipped = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            url = _extract_image_url(part)
            if not url:
                continue

            seen += 1
            try:
                image_bytes = _read_image_bytes(url)
                data_url, old_size, new_size = _resize_image_to_data_url(
                    image_bytes, MAX_IMAGE_SIDE
                )
                if data_url is None:
                    skipped += 1
                    continue

                _set_image_url(part, data_url)
                resized += 1
                log.info(
                    "Job %s: resized image %s from %sx%s to %sx%s",
                    job_id,
                    seen,
                    old_size[0],
                    old_size[1],
                    new_size[0],
                    new_size[1],
                )
            except Exception as exc:
                skipped += 1
                log.warning("Job %s: image resize skipped (%s)", job_id, exc)

    if seen:
        log.info(
            "Job %s: image preprocessing complete (seen=%s resized=%s skipped=%s max_side=%s)",
            job_id,
            seen,
            resized,
            skipped,
            MAX_IMAGE_SIDE,
        )


def handler(job):
    """
    RunPod handler. Forwards the job input directly to vLLM's
    OpenAI-compatible chat completions endpoint.

    Expected input format (same as OpenAI chat completions):
    {
        "model": "zai-org/GLM-OCR",
        "messages": [...],
        "max_tokens": 2048,
        "temperature": 0.0
    }
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input")
    if isinstance(job_input, dict):
        job_input = dict(job_input)
    elif isinstance(job_input, str):
        pass
    else:
        return {
            "error": (
                "Invalid request format. Expected {'input': {...}} or "
                "{'input': '<image_url_or_path>'}."
            )
        }
    log.info("Job %s: received request", job_id)

    if VLLM_STARTUP_ERROR:
        log.error("Job %s: refusing request because vLLM is unavailable", job_id)
        return _vllm_error_payload(
            job_id,
            "Local vLLM server failed to initialize.",
        )

    if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is not None:
        message = (
            f"Local vLLM process exited with code {VLLM_PROCESS.returncode}. "
            f"GPU diagnostics: {_format_gpu_diagnostics()}"
        )
        _set_vllm_startup_error(message)
        log.error("Job %s: %s", job_id, message)
        return _vllm_error_payload(
            job_id,
            "Local vLLM process is not running.",
        )

    sdk_result = _parse_with_sdk(job_input, job_id)
    if sdk_result is not None:
        log.info("Job %s: completed via glm-ocr SDK", job_id)
        return sdk_result

    if not isinstance(job_input, dict):
        return {
            "error": (
                "No image found for glm-ocr SDK parse and input is not a "
                "chat completions payload."
            )
        }

    # Set model default if not provided.
    if "model" not in job_input:
        job_input["model"] = MODEL_NAME

    # vLLM chat completions requires messages.
    if "messages" not in job_input:
        log.error("Job %s: missing required 'messages' field", job_id)
        return {
            "error": "Input must contain a 'messages' field for Chat Completions API."
        }

    preprocess_images(job_input, job_id)

    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=job_input,
            timeout=600,
        )

        if response.status_code != 200:
            body = _truncate_text(response.text, 4000)
            log.error(
                "Job %s: vLLM returned %s with body: %s",
                job_id,
                response.status_code,
                body,
            )
            return _vllm_error_payload(
                job_id,
                f"Local vLLM server returned HTTP {response.status_code}.",
                http_status=response.status_code,
                response_body=body,
            )
        result = response.json()
        log.info("Job %s: completed", job_id)
        return result
    except requests.exceptions.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" | response_body={_truncate_text(exc.response.text, 4000)}"
        log.error("Job %s: failed - %s%s", job_id, exc, detail)
        return _vllm_error_payload(
            job_id,
            "Request to local vLLM server failed.",
            request_error=str(exc),
            response_body=_truncate_text(
                getattr(getattr(exc, "response", None), "text", ""),
                4000,
            ),
        )


if __name__ == "__main__":
    VLLM_PROCESS = start_vllm()
    try:
        wait_for_vllm(VLLM_PROCESS)
    except Exception as exc:
        _set_vllm_startup_error(str(exc))
        log.error("Worker startup aborted: %s", exc)
        sys.exit(1)
    OCR_PARSER = init_glmocr_sdk()
    runpod.serverless.start({"handler": handler})
