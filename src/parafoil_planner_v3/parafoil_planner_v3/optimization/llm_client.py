from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


def _get_env(keys: list[str], default: str | None = None) -> str | None:
    for k in keys:
        if k in os.environ and os.environ[k]:
            return os.environ[k]
    return default


def call_llm(
    prompt: str,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> str:
    """
    Call an OpenAI-compatible chat completion endpoint.
    Uses env: LLM_API_BASE/OPENAI_API_BASE, LLM_API_KEY/OPENAI_API_KEY, LLM_MODEL/OPENAI_MODEL.
    """
    base = api_base or _get_env(["LLM_API_BASE", "OPENAI_API_BASE"], "https://api.openai.com/v1")
    key = api_key or _get_env(["LLM_API_KEY", "OPENAI_API_KEY"], "")
    model = model or _get_env(["LLM_MODEL", "OPENAI_MODEL"], "gpt-4o-mini")
    if not key:
        raise RuntimeError("LLM_API_KEY/OPENAI_API_KEY not set")
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    choices = obj.get("choices", []) if isinstance(obj, dict) else []
    if not choices:
        raise RuntimeError("LLM response missing choices")
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    return str(msg.get("content", "")).strip()


def extract_yaml_block(text: str) -> dict[str, Any] | None:
    """
    Extract first YAML block from response. Accepts fenced ```yaml or plain code fence.
    """
    patterns = [
        r"```yaml\s*(.*?)```",
        r"```yml\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            block = m.group(1)
            try:
                import yaml

                obj = yaml.safe_load(block)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
    # Fallback: try parse full text
    try:
        import yaml

        obj = yaml.safe_load(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None
