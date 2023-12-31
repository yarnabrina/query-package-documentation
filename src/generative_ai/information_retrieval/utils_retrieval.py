import enum
import typing

import pydantic
import typing_extensions
from langchain.vectorstores.chroma import Chroma


class RetrievalType(str, enum.Enum):
    MMR = "mmr"
    SIMILARITY = "similarity"


class TransformerType(str, enum.Enum):
    STANDARD_TRANSFORMERS = "standard_transformers"
    QUANTISED_CTRANSFORMERS = "quantised_ctransformers"


class PipelineType(str, enum.Enum):
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"


class StandardModel(pydantic.BaseModel):
    language_model_type: typing.Literal[TransformerType.STANDARD_TRANSFORMERS]
    standard_pipeline_type: PipelineType
    standard_model_name: str


class QuantisedModel(pydantic.BaseModel):
    language_model_type: typing.Literal[TransformerType.QUANTISED_CTRANSFORMERS]
    quantised_model_name: str
    quantised_model_file: str
    quantised_model_type: str


LanguageModel = typing_extensions.TypeAliasType(
    "LanguageModel",
    typing.Annotated[
        QuantisedModel | StandardModel, pydantic.Field(discriminator="language_model_type")
    ],
)
LanguageModelAdapter = pydantic.TypeAdapter(LanguageModel)

ValidatedChroma = pydantic.InstanceOf[Chroma]


__all__ = [
    "LanguageModel",
    "LanguageModelAdapter",
    "TransformerType",
    "PipelineType",
    "QuantisedModel",
    "RetrievalType",
    "StandardModel",
    "ValidatedChroma",
]
