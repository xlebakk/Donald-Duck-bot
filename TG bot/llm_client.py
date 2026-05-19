"""
llm_client.py — единый интерфейс для Ollama и LM Studio.
"""

from __future__ import annotations

import json
import httpx
from typing import AsyncIterator

import config


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _build_messages(history: list[dict]) -> list[dict]:
    """Добавляет system-prompt в начало истории, если его ещё нет."""
    if getattr(config, "SYSTEM_PROMPT", None) and (not history or history[0]["role"] != "system"):
        return [{"role": "system", "content": config.SYSTEM_PROMPT}] + history
    return history


# ─────────────────────────────────────────────
#  Ollama  (http://localhost:11434)
# ─────────────────────────────────────────────

class OllamaClient:
    """Работает через /api/chat с поддержкой стриминга."""

    def __init__(self) -> None:
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model = config.OLLAMA_MODEL

    async def chat_stream(self, history: list[dict]) -> AsyncIterator[str]:
        # Для Ollama настройки (options) передаются отдельным словарем
        temp = getattr(config, "TEMPERATURE", 1.2)
        payload = {
            "model": self.model,
            "messages": _build_messages(history),
            "stream": True,
            "options": {
                "temperature": float(temp)
            }
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break

    async def chat(self, history: list[dict]) -> str:
        result = []
        async for chunk in self.chat_stream(history):
            result.append(chunk)
        return "".join(result)


# ─────────────────────────────────────────────
#  LM Studio  (http://localhost:1234/v1)
# ─────────────────────────────────────────────

class LMStudioClient:
    """Работает через OpenAI-совместимый /v1/chat/completions."""

    def __init__(self) -> None:
        self.base_url = config.LMSTUDIO_BASE_URL.rstrip("/")
        self.model = config.LMSTUDIO_MODEL

    async def chat_stream(self, history: list[dict]) -> AsyncIterator[str]:
        # Извлекаем температуру из config, если её там нет — ставим взрывные 1.2
        temp = getattr(config, "TEMPERATURE", 1.2)
        
        payload = {
            "model": self.model,
            "messages": _build_messages(history),
            "stream": True,
            "temperature": float(temp),
        }
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    delta = data["choices"][0].get("delta", {})
                    chunk = delta.get("content", "")
                    if chunk:
                        yield chunk

    async def chat(self, history: list[dict]) -> str:
        result = []
        async for chunk in self.chat_stream(history):
            result.append(chunk)
        return "".join(result)


# ─────────────────────────────────────────────
#  Фабрика — возвращает нужный клиент
# ─────────────────────────────────────────────

def get_llm_client() -> OllamaClient | LMStudioClient:
    backend = config.LLM_BACKEND.lower()
    if backend == "ollama":
        return OllamaClient()
    elif backend == "lmstudio":
        return LMStudioClient()
    else:
        raise ValueError(
            f"Неизвестный LLM_BACKEND: '{backend}'. "
            "Используй 'ollama' или 'lmstudio'."
        )