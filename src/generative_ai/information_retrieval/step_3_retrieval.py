import time
import typing
import uuid

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class CaptureDetailsCallback(BaseCallbackHandler):
    def __init__(self: "CaptureDetailsCallback") -> None:
        super().__init__()

        self.effective_prompt: str | None = None
        self.effective_duration: float | None = None

    def on_llm_start(  # noqa: PLR0913
        self: "CaptureDetailsCallback",
        serialized: dict,
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        **kwargs: typing.Any,
    ) -> None:
        del serialized
        del run_id
        del parent_run_id
        del tags
        del metadata
        del kwargs

        self.effective_prompt = prompts[0]
        self.effective_duration = time.perf_counter()

    def on_llm_end(
        self: "CaptureDetailsCallback",
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: typing.Any,
    ) -> None:
        del response
        del run_id
        del parent_run_id
        del kwargs

        self.effective_duration = time.perf_counter() - self.effective_duration


__all__ = ["CaptureDetailsCallback"]
