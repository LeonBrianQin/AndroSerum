# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, Optional

from params import LLMParams


class LLMError(RuntimeError):
    pass


class RetriableLLMError(LLMError):
    pass


class OpenAIClient:
    """Minimal OpenAI REST client with robust error handling.

    - Uses Responses API (/v1/responses) by default.
    - Supports direct api_key from config (preferred) or env var name.
    - Fails fast on insufficient_quota (non-retriable).
    """

    def __init__(self, params: LLMParams):
        self.p = params
        key = (self.p.api_key or "").strip()
        if not key:
            env_name = (self.p.api_key_env or "").strip()
            if env_name:
                key = (os.getenv(env_name, "") or "").strip()
        if not key:
            raise LLMError(
                "Missing API key. Provide CONFIG['llm_api_key'] "
                "or set the environment variable named by CONFIG['llm_api_key_env']."
            )
        self.api_key = key

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        if (self.p.provider or "").lower() == "openrouter":
            if (self.p.site_url or "").strip():
                h["HTTP-Referer"] = self.p.site_url.strip()
            if (self.p.app_name or "").strip():
                h["X-OpenRouter-Title"] = self.p.app_name.strip()
        return h

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        last_err: Optional[str] = None
        for attempt in range(1, self.p.max_retries + 1):
            try:
                # ---- proxy (important) ----
                proxy = (self.p.proxy or "").strip()
                if not proxy:
                    # fallback: env vars
                    proxy = (
                            os.environ.get("https_proxy")
                            or os.environ.get("HTTPS_PROXY")
                            or os.environ.get("http_proxy")
                            or os.environ.get("HTTP_PROXY")
                            or ""
                    ).strip()

                proxies = {"http": proxy, "https": proxy} if proxy else None
                if attempt == 1:
                    print(f"[LLM] proxy={proxy!r} proxies_enabled={bool(proxies)}")

                r = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=(self.p.connect_timeout_sec, self.p.read_timeout_sec),
                    proxies=proxies,
                )

                if 200 <= r.status_code < 300:
                    return r.json()

                # parse error payload if possible
                err_obj: Dict[str, Any] = {}
                try:
                    err_obj = (r.json() or {}).get("error") or {}
                except Exception:
                    err_obj = {}
                code = err_obj.get("code") or err_obj.get("type")

                # non-retriable
                if r.status_code == 429 and code == "insufficient_quota":
                    raise LLMError(f"Insufficient quota: {err_obj}")
                if r.status_code in (400, 401, 403, 404):
                    raise LLMError(f"HTTP {r.status_code}: {err_obj or r.text[:400]}")

                # retriable bucket
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {err_obj or r.text[:400]}"
                    sleep = (self.p.retry_base_sec ** attempt) + random.random()
                    time.sleep(min(20.0, sleep))
                    continue

                raise LLMError(f"HTTP {r.status_code}: {err_obj or r.text[:400]}")

            except requests.Timeout as e:
                last_err = f"timeout: {e}"
                sleep = (self.p.retry_base_sec ** attempt) + random.random()
                time.sleep(min(20.0, sleep))
                continue
            except LLMError:
                raise
            except Exception as e:
                last_err = f"exception: {e}"
                sleep = (self.p.retry_base_sec ** attempt) + random.random()
                time.sleep(min(20.0, sleep))
                continue

        raise RetriableLLMError(last_err or "unknown error")

    def responses(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        base = self.p.base_url.rstrip("/")
        url = base + "/responses"  # base already contains /v1
        payload = {
            "model": self.p.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": self.p.temperature,
        }
        return self._post_json(url, payload)

    def chat_completions(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        base = self.p.base_url.rstrip("/")
        url = base + "/chat/completions"
        payload = {
            "model": self.p.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_output_tokens,
            "temperature": self.p.temperature,
        }
        return self._post_json(url, payload)
