import transformers
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.vectorstores.chroma import Chroma


def create_database_retriever(embedding_database: Chroma) -> VectorStoreRetriever:
    retriever = embedding_database.as_retriever(search_kwargs={"k": 3})

    return retriever


def create_llm(language_model: str) -> HuggingFacePipeline:
    pipeline = transformers.pipeline(model=language_model, use_fast=True, max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=pipeline)

    return llm


def generate_chain(
    database_retriever: VectorStoreRetriever, llm: HuggingFacePipeline
) -> BaseRetrievalQA:
    prompt_template = """
    You are an assistant for question-answering tasks.

    Your goal is to answer the following question delimited by triple single quotes.

    Question: '''{question}'''

    Only use the following context delimited by triple single quotes to answer the above question.

    Context: '''{context}'''

    Keep the answer concise, and use at most three sentences.

    If you do not know the answer, do not make up any information. Just reply 'Unable to answer'.

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    retrieval_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs={"prompt": prompt},
        retriever=database_retriever,
        return_source_documents=True,
    )

    return retrieval_chain


__all__ = ["create_database_retriever", "create_llm", "generate_chain"]
