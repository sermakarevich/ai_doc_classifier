from __future__ import annotations

import base64
import json
import logging
import os
from typing import TypeVar

import httpx
from pydantic import BaseModel, model_validator
from pydantic import ValidationError as PydanticValidationError

from .constants import (
    DEFAULT_BASE_URL,
    DEFAULT_PROVIDERS,
    MAX_ATTEMPTS,
    PROVIDERS_ENV_VAR,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


class ProviderConfig(BaseModel):
    model: str
    name: str = ""
    base_url: str = DEFAULT_BASE_URL

    @model_validator(mode="before")
    @classmethod
    def _default_name(cls, data: object) -> object:
        if isinstance(data, dict) and "name" not in data and "model" in data:
            data["name"] = data["model"]
        return data

    @property
    def display_name(self) -> str:
        return self.name or self.model


class OllamaProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.display_name

    @property
    def base_url(self) -> str:
        return self._config.base_url

    async def structured(
        self, prompt: str, response_model: type[T], images: list[bytes] | None = None
    ) -> T:
        message: dict[str, object] = {"role": "user", "content": prompt}
        if images:
            message["images"] = [base64.b64encode(img).decode("ascii") for img in images]

        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": [message],
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {"temperature": TEMPERATURE},
        }

        last_err: Exception = ValueError("unexpected error")

        for attempt in range(MAX_ATTEMPTS):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=httpx.Timeout(None),
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    content = body["message"]["content"]
                    return response_model.model_validate_json(content)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, PydanticValidationError) as err:
                last_err = err
                logger.warning("Attempt %d failed for provider %s: %s", attempt + 1, self.name, err)
                if attempt < MAX_ATTEMPTS - 1:
                    logger.info("Retrying once for %s", self.name)
                    continue
                break

        raise StructuredOutputError(str(last_err))


def load_providers(env_var: str = PROVIDERS_ENV_VAR) -> list[OllamaProvider]:
    raw = os.environ.get(env_var)
    if raw is None:
        dicts = DEFAULT_PROVIDERS
    else:
        dicts = json.loads(raw)

    return [OllamaProvider(ProviderConfig(**d)) for d in dicts]
