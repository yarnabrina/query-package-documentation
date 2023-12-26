import pydantic


class Response(pydantic.BaseModel):
    query: str
    answer: str
    source_documents: list[str]


__all__ = ["Response"]
