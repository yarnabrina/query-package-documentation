"""Define functionalities for type annotations for top level modules."""

import pydantic


class Response(pydantic.BaseModel):
    """Response from large language model with additional captured details.

    Attributes
    ----------
    query : str
        query from user
    answer : str
        response from large language model
    source_documents : list[str]
        list of source documents retrieved from database
    used_prompt : str
        exact prompt passed to large language model
    llm_duration : float
        time taken (in seconds) for large language model to generate response
    """

    query: str
    answer: str
    source_documents: list[str]
    used_prompt: str
    llm_duration: float


__all__ = ["Response"]
