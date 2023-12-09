import transformers
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms.ctransformers import CTransformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.vectorstores.chroma import Chroma

from .utils_retrieval import LLAMA2_MODEL, MISTRAL_MODEL, ZEPHYR_MODEL, LanguageModelType


def create_database_retriever(embedding_database: Chroma) -> VectorStoreRetriever:
    retriever = embedding_database.as_retriever(search_kwargs={"k": 10})

    return retriever


def create_llm(
    language_model_type: LanguageModelType, language_model_name: str
) -> CTransformers | HuggingFacePipeline:
    match language_model_type:
        case LanguageModelType.HUGGINGFACE_STANDARD:
            tokeniser = transformers.AutoTokenizer.from_pretrained(
                language_model_name, use_fast=True, padding="max_length", truncation=True
            )
            tokeniser.pad_token = tokeniser.eos_token

            pipeline = transformers.pipeline(
                model=language_model_name, tokenizer=tokeniser, max_new_tokens=300
            )
            llm = HuggingFacePipeline(pipeline=pipeline)
        case LanguageModelType.LLAMA2_7B_GGUF:
            llm = CTransformers(
                model=LLAMA2_MODEL.model,
                model_type=LLAMA2_MODEL.model_type,
                model_file=LLAMA2_MODEL.model_file,
            )
        case LanguageModelType.MISTRAL_7B_GGUF:
            llm = CTransformers(
                model=MISTRAL_MODEL.model,
                model_type=MISTRAL_MODEL.model_type,
                model_file=MISTRAL_MODEL.model_file,
            )
        case LanguageModelType.ZEPHYR_7B_GGUF:
            llm = CTransformers(
                model=ZEPHYR_MODEL.model,
                model_type=ZEPHYR_MODEL.model_type,
                model_file=ZEPHYR_MODEL.model_file,
            )
        case _:
            raise ValueError("Unexpected language model type")

    return llm


def generate_retrieval_chain(
    database_retriever: VectorStoreRetriever,
    llm: CTransformers | HuggingFacePipeline,
    language_model_type: LanguageModelType,
) -> BaseRetrievalQA:
    match language_model_type:
        case LanguageModelType.HUGGINGFACE_STANDARD | LanguageModelType.LLAMA2_7B_GGUF:
            prompt_template = """You are a chat assistant for question-answering tasks.

You help human developers with documentation on a new python package they are unfamiliar with.

Your goal is to answer the following question, delimited by triple single quotes.

Question: '''{question}'''

Only use the following context delimited by triple single quotes to answer the above question."

Context: '''{context}'''

Keep the answer as concise as possible, and do not use more than five sentences.

If you do not know the answer, just reply 'Sorry, I do not know.'.

Do not make up any information.

Answer: """
        case LanguageModelType.MISTRAL_7B_GGUF | LanguageModelType.ZEPHYR_7B_GGUF:
            prompt_template = """[INST]

<<SYS>> You are a chat assistant for question-answering tasks.

You help human developers with documentation on a new python package they are unfamiliar with.

You may use the retrieved context to answer the question, both delimited by triple single quotes.

Keep the answer as concise as possible, and do not use more than five sentences.

If you do not know the answer, just reply 'Sorry, I do not know.'.

Do not use any information that is not available in the context.

Do not make up any information. <</SYS>>

Question: '''{question}'''

Context: '''{context}'''

Answer: [/INST]"""
        case _:
            raise ValueError("Unexpected language model type")

    prompt = PromptTemplate.from_template(prompt_template)

    retrieval_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs={"prompt": prompt},
        retriever=database_retriever,
        return_source_documents=True,
    )

    return retrieval_chain


__all__ = ["create_database_retriever", "create_llm", "generate_retrieval_chain"]
