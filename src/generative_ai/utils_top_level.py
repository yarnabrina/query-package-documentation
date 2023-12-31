import pydantic


class Response(pydantic.BaseModel):
    query: str
    answer: str
    source_documents: list[str]
    used_prompt: str
    llm_duration: float


__all__ = ["Response"]
