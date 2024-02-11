"""Define functionalities for type annotations in information retrieval step."""

import enum
import typing

import pydantic


class RetrievalType(str, enum.Enum):
    """Define supported retrieval types."""

    MMR = "mmr"
    SIMILARITY = "similarity"


class TransformerType(str, enum.Enum):
    """Define supported transformer types."""

    STANDARD_TRANSFORMERS = "standard_transformers"
    QUANTISED_CTRANSFORMERS = "quantised_ctransformers"


class PipelineType(str, enum.Enum):
    """Define supported pipeline types."""

    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"


class StandardModel(pydantic.BaseModel):
    """Store details of a ``transformers`` library compatible Hugging Face model.

    Attributes
    ----------
    language_model_type : typing.Literal[TransformerType.STANDARD_TRANSFORMERS]
        kind of language model
    standard_pipeline_type : PipelineType
        kind of Hugging Face pipeline
    standard_model_name : str
        name of the Hugging Face model
    """

    language_model_type: typing.Literal[TransformerType.STANDARD_TRANSFORMERS]
    standard_pipeline_type: PipelineType
    standard_model_name: str


class QuantisedModel(pydantic.BaseModel):
    """Store details of a ``ctransformers`` library compatible Hugging Face model.

    Attributes
    ----------
    language_model_type : typing.Literal[TransformerType.QUANTISED_CTRANSFORMERS]
        kind of language model
    quantised_model_name : str
        name of the Hugging Face model
    quantised_model_file : str
        named of quantised model file
    quantised_model_type : str
        type of quantised model
    """

    language_model_type: typing.Literal[TransformerType.QUANTISED_CTRANSFORMERS]
    quantised_model_name: str
    quantised_model_file: str
    quantised_model_type: str


LanguageModel = typing.Annotated[
    QuantisedModel | StandardModel, pydantic.Field(discriminator="language_model_type")
]
LanguageModelAdapter = pydantic.TypeAdapter(LanguageModel)


__all__ = [
    "LanguageModel",
    "LanguageModelAdapter",
    "TransformerType",
    "PipelineType",
    "QuantisedModel",
    "RetrievalType",
    "StandardModel",
]
