import transformers
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms.ctransformers import CTransformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStoreRetriever

from .utils_retrieval import (
    LLAMA2_MODEL,
    MISTRAL_MODEL,
    ZEPHYR_MODEL,
    LanguageModelType,
    ValidatedChroma,
)


def create_database_retriever(
    embedding_database: ValidatedChroma,
) -> VectorStoreRetriever:
    retriever = embedding_database.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )

    return retriever


def create_llm(
    language_model_type: LanguageModelType, language_model_name: str
) -> CTransformers | HuggingFacePipeline:
    common_parameters = {"max_new_tokens": 256}

    match language_model_type:
        case LanguageModelType.HUGGINGFACE_STANDARD:
            common_parameters.update({"do_sample": True, "top_k": 1})
        case _:
            common_parameters.update({"temperature": 0})

    match language_model_type:
        case LanguageModelType.HUGGINGFACE_STANDARD:
            tokeniser = transformers.AutoTokenizer.from_pretrained(
                language_model_name, use_fast=True, padding="max_length", truncation=True
            )
            tokeniser.pad_token = tokeniser.eos_token

            pipeline = transformers.pipeline(
                model=language_model_name, tokenizer=tokeniser, **common_parameters
            )
            llm = HuggingFacePipeline(pipeline=pipeline)
        case LanguageModelType.LLAMA2_7B_GGUF:
            llm = CTransformers(
                model=LLAMA2_MODEL.model,
                model_type=LLAMA2_MODEL.model_type,
                model_file=LLAMA2_MODEL.model_file,
                config=common_parameters,
            )
        case LanguageModelType.MISTRAL_7B_GGUF:
            llm = CTransformers(
                model=MISTRAL_MODEL.model,
                model_type=MISTRAL_MODEL.model_type,
                model_file=MISTRAL_MODEL.model_file,
                config=common_parameters,
            )
        case LanguageModelType.ZEPHYR_7B_GGUF:
            llm = CTransformers(
                model=ZEPHYR_MODEL.model,
                model_type=ZEPHYR_MODEL.model_type,
                model_file=ZEPHYR_MODEL.model_file,
                config=common_parameters,
            )
        case _:
            raise ValueError("Unexpected language model type")

    return llm


def generate_retrieval_chain(
    database_retriever: VectorStoreRetriever, llm: CTransformers | HuggingFacePipeline
) -> BaseRetrievalQA:
    prompt_template = """You are a chat assistant to help new users for a Python package.

1. You will be provided with a specific question and a context relevant to answer that question.
2. Your response should be based solely on the given context.
3. Keep your answer concise, not exceeding five sentences.
4. If the answer is not found within the context, respond with "I do not know.".
5. Do not fabricate any information.

Context: {context}
Question: {question}

Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)

    retrieval_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs={"prompt": prompt},
        retriever=database_retriever,
        return_source_documents=True,
    )

    return retrieval_chain


__all__ = ["create_database_retriever", "create_llm", "generate_retrieval_chain"]
