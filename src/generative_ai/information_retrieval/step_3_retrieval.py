"""Define functionalities to debug large language model generation process."""

import time
import typing

from langchain.callbacks.base import BaseCallbackHandler

if typing.TYPE_CHECKING:
    import uuid

    from langchain_core.outputs import LLMResult


class CaptureDetailsCallback(BaseCallbackHandler):
    """Capture details of question answering pipeline.

    Attributes
    ----------
    effective_prompt : str | None
        exact prompt passed to large language model after successful retrieval
    effective_duration : float | None
        time taken (in seconds) for large language model to generate response
    """

    def __init__(self: "CaptureDetailsCallback") -> None:
        super().__init__()

        self.effective_prompt: str | None = None
        self.effective_duration: float | None = None

    def on_llm_start(  # noqa: PLR0913, numpydoc ignore=PR01
        self: "CaptureDetailsCallback",
        serialized: dict,
        prompts: list[str],
        *,
        run_id: "uuid.UUID",
        parent_run_id: "uuid.UUID | None" = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        **kwargs: typing.Any,  # noqa: ANN401
    ) -> None:
        """Run when large language model starts generating response.

        Notes
        -----
        * This method only uses ``prompts`` argument, and rest are ignored.
        * This modifies ``self.effective_prompt`` and ``self.effective_duration`` attributes.

            * ``self.effective_prompt`` is set to the first element of ``prompts``.
            * ``self.effective_duration`` is set to the current time.
        """
        del serialized
        del run_id
        del parent_run_id
        del tags
        del metadata
        del kwargs

        self.effective_prompt = prompts[0]
        self.effective_duration = time.perf_counter()

    def on_llm_end(  # numpydoc ignore=PR01
        self: "CaptureDetailsCallback",
        response: "LLMResult",
        *,
        run_id: "uuid.UUID",
        parent_run_id: "uuid.UUID | None" = None,
        **kwargs: typing.Any,  # noqa: ANN401
    ) -> None:
        """Run when large language model finishes generating response.

        Notes
        -----
        * This method ignores all of its arguments.
        * This modifies ``self.effective_duration`` attribute.

            * It is updated to the difference between current time and the stored value.
        """
        del response
        del run_id
        del parent_run_id
        del kwargs

        self.effective_duration = time.perf_counter() - self.effective_duration


__all__ = ["CaptureDetailsCallback"]
