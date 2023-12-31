import transformers
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms.ctransformers import CTransformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStoreRetriever

from .utils_retrieval import LanguageModel, RetrievalType, TransformerType, ValidatedChroma


def create_database_retriever(
    embedding_database: ValidatedChroma,
    search_type: RetrievalType,
    number_of_documents: int,
    initial_number_of_documents: int,
    diversity_level: float,
) -> VectorStoreRetriever:
    retriever = embedding_database.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": number_of_documents,
            "fetch_k": initial_number_of_documents,
            "lambda_mult": diversity_level,
        },
    )

    return retriever


def create_llm(language_model: LanguageModel) -> CTransformers | HuggingFacePipeline:
    common_parameters = {"max_new_tokens": 256}

    match language_model.language_model_type:
        case TransformerType.STANDARD_TRANSFORMERS:
            common_parameters.update({"do_sample": True, "top_k": 1})

            tokeniser = transformers.AutoTokenizer.from_pretrained(
                language_model.standard_model_name,
                use_fast=True,
                padding="max_length",
                truncation=True,
            )
            tokeniser.pad_token = tokeniser.eos_token

            pipeline = transformers.pipeline(
                task=language_model.standard_pipeline_type,
                model=language_model.standard_model_name,
                tokenizer=tokeniser,
                model_kwargs={"low_cpu_mem_usage": True},
                **common_parameters,
            )

            llm = HuggingFacePipeline(pipeline=pipeline)
        case TransformerType.QUANTISED_CTRANSFORMERS:
            common_parameters.update({"temperature": 0})

            llm = CTransformers(
                model=language_model.quantised_model_name,
                model_type=language_model.quantised_model_type,
                model_file=language_model.quantised_model_file,
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

Context

{context}

Question

{question}

Answer

"""

    prompt = PromptTemplate.from_template(prompt_template)

    retrieval_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs={"prompt": prompt},
        retriever=database_retriever,
        return_source_documents=True,
    )

    return retrieval_chain


__all__ = ["create_database_retriever", "create_llm", "generate_retrieval_chain"]
