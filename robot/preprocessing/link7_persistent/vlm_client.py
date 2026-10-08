"""Backend-neutral local-GPU and cloud-API vision-language model client."""
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

from infrastructure.shared.contracts.vlm_failure import VLMTimeoutError, is_request_timeout


class MissingVLMTextError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("VLM response is missing text: " + json.dumps(diagnostic, sort_keys=True))


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _response_text(data):
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    return "".join(
        part.get("text", "")
        for item in data.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    )


class VLMClient:
    def __init__(self, role, config):
        self.role = role
        self.config = dict(config)
        self.backend = self.config["backend"]
        self.model = str(self.config["model"])
        self.records = []
        if self.backend == "local_gpu":
            from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import load_model
            self.processor, self.local_model, self.architecture = load_model(self.model)
        elif self.backend != "cloud_api":
            raise ValueError(f"Unsupported backend: {self.backend}")

    def ask(self, images, user_prompt, *, system_prompt=""):
        images = list(images)
        payloads = [_png(image) for image in images]
        record = {
            "role": self.role,
            "backend": self.backend,
            "requested_model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_sha256": [hashlib.sha256(data).hexdigest() for data in payloads],
        }
        self.records.append(record)
        if self.backend == "local_gpu":
            answer = self._ask_local(images, system_prompt, user_prompt)
            record.update(answer=answer, model=self.model, architecture=self.architecture)
            return record
        try:
            answer, metadata = self._ask_cloud(payloads, system_prompt, user_prompt)
        except MissingVLMTextError as exc:
            record["response_diagnostic"] = exc.diagnostic
            raise
        record.update(metadata, answer=answer)
        return record

    def _ask_local(self, images, system_prompt, user_prompt):
        from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import generate
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": content})
        return generate(self.local_model, self.processor, messages, int(self.config.get("max_tokens", 800)))

    def _ask_cloud(self, payloads, system_prompt, user_prompt):
        style = self.config["api_style"]
        base = str(self.config["api_base"]).rstrip("/")
        encoded = ["data:image/png;base64," + base64.b64encode(data).decode("ascii") for data in payloads]
        if style == "responses":
            content = [{"type": "input_text", "text": user_prompt}]
            content.extend({"type": "input_image", "detail": "original", "image_url": url} for url in encoded)
            payload = {"model": self.model, "instructions": system_prompt,
                       "reasoning": {"effort": self.config.get("reasoning_effort", "none")},
                       "store": False, "input": [{"role": "user", "content": content}]}
            url = base + "/responses"
        else:
            content = [{"type": "image_url", "image_url": {"url": url}} for url in encoded]
            content.append({"type": "text", "text": user_prompt})
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})
            payload = {"model": self.model, "messages": messages}
            temperature = self.config.get("temperature", 0)
            if temperature is not None:
                payload["temperature"] = temperature
            if "max_completion_tokens" in self.config:
                payload["max_completion_tokens"] = int(self.config["max_completion_tokens"])
            else:
                payload["max_tokens"] = int(self.config.get("max_tokens", 800))
            effort = self.config.get("reasoning_effort")
            if effort:
                payload["reasoning_effort"] = effort
            url = base + "/chat/completions"
        key = os.environ[self.config["api_key_env"]]
        timeout = int(self.config.get("timeout_seconds", 240))
        attempts = int(os.environ.get('PDI_VLM_TRANSPORT_ATTEMPTS', '3'))
        if not 1 <= attempts <= 3:
            raise ValueError('Transport attempts must be 1..3')
        for attempt in range(attempts):
            request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = json.load(response)
                if style == "responses":
                    answer = _response_text(result)
                else:
                    answer = result["choices"][0].get("message", {}).get("content", "")
                    if isinstance(answer, list):
                        answer = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in answer)
                if not isinstance(answer, str) or not answer:
                    choice = result.get("choices", [{}])[0] if style != "responses" else {}
                    message = choice.get("message") or {}
                    raise MissingVLMTextError({
                        "model": result.get("model"),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": result.get("usage"),
                        "status": result.get("status"),
                        "content_type": type(message.get("content")).__name__,
                        "message_fields": sorted(message),
                    })
                metadata = {name: result.get(name) for name in ("id", "model", "provider", "status", "usage")}
                return answer, metadata
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                    try:
                        detail = json.loads(error.read()).get("error", {}).get("message", "Request rejected")
                    except (ValueError, AttributeError):
                        detail = "Request rejected"
                    detail = str(detail).replace(key, "[REDACTED]")
                    raise RuntimeError(f"VLM cloud API HTTP {error.code}: {detail}") from None
            except (TimeoutError, urllib.error.URLError) as error:
                if is_request_timeout(error):
                    raise VLMTimeoutError("VLM request timed out without a response") from None
                if attempt == attempts - 1:
                    raise RuntimeError("VLM cloud API request failed") from None
            time.sleep(2**attempt)
        raise RuntimeError("VLM request exhausted")


class VisionRoute:
    """Compatibility adapter for retained mask-refinement experiments."""
    def __init__(self, provider, model):
        if provider == "qwen":
            config = {"backend": "local_gpu", "model": str(model), "max_tokens": 1800}
        elif provider == "openrouter":
            config = {"backend": "cloud_api", "model": str(model), "api_style": "chat_completions",
                      "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
                      "reasoning_effort": "none", "max_tokens": 1800, "timeout_seconds": 180}
        else:
            raise ValueError("Unknown VLM provider")
        self.client = VLMClient("legacy_refinement", config)
        self.records = self.client.records

    def ask(self, images, prompt):
        return self.client.ask(images, prompt)["answer"]
