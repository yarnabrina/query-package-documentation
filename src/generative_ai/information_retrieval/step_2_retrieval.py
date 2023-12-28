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
    RetrievalType,
    ValidatedChroma,
)


def create_database_retriever(
    embedding_database: ValidatedChroma,
    search_type: RetrievalType,
    number_of_documents: int,
    number_of_diverse_documents: int,
) -> VectorStoreRetriever:
    retriever = embedding_database.as_retriever(
        search_type=search_type,
        search_kwargs={"k": number_of_documents, "fetch_k": number_of_diverse_documents},
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
    prompt_template = """You are a chat assistant for question answering tasks.

Use the following retrieved context to answer the given question.

If the answer is not in the context, say "I do not know.".

Keep your answer as concise as possible.

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
