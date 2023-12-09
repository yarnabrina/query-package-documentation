import enum
import typing


class LanguageModelType(str, enum.Enum):
    HUGGINGFACE_STANDARD = "huggingface_standard"
    LLAMA2_7B_GGUF = "llama2_7b_gguf"
    MISTRAL_7B_GGUF = "mistral_7b_gguf"
    ZEPHYR_7B_GGUF = "zephyr_7b_gguf"


class QuantisedModel(typing.NamedTuple):
    model: str
    model_file: str
    model_type: str


LLAMA2_MODEL = QuantisedModel("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf", "llama")
MISTRAL_MODEL = QuantisedModel(
    "TheBloke/Mistral-7B-v0.1-GGUF", "mistral-7b-v0.1.Q4_K_M.gguf", "mistral"
)
ZEPHYR_MODEL = QuantisedModel(
    "TheBloke/zephyr-7B-beta-GGUF", "zephyr-7b-beta.Q4_K_M.gguf", "mistral"
)


__all__ = ["LLAMA2_MODEL", "MISTRAL_MODEL", "ZEPHYR_MODEL", "LanguageModelType", "QuantisedModel"]
