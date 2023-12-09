import pathlib

from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSerializable
from langchain.vectorstores.chroma import Chroma

from .step_1_retrieval import (
    create_document_embedder,
    create_vector_store,
    load_json_documents,
    partition_documents,
)
from .step_2_retrieval import create_database_retriever, create_llm, generate_retrieval_chain
from .utils_retrieval import LanguageModelType


def load_source_documents(file_path: pathlib.Path) -> list[Document]:
    raw_documents = load_json_documents(file_path)
    partitioned_documents = partition_documents(raw_documents)

    return partitioned_documents


def create_embedding_database(
    embedding_model: str, directory_path: pathlib.Path, source_documents: list[Document]
) -> Chroma:
    document_embedder = create_document_embedder(embedding_model)

    vector_store = create_vector_store(document_embedder, directory_path)
    vector_store.add_documents(source_documents)

    return vector_store


def store_embedding_database(vector_store: Chroma) -> None:
    vector_store.persist()


def load_embedding_database(embedding_model: str, directory_path: pathlib.Path) -> Chroma:
    document_embedder = create_document_embedder(embedding_model)

    vector_store = create_vector_store(document_embedder, directory_path)

    return vector_store


def prepare_question_answer_chain(
    embedding_database: Chroma, language_model_type: LanguageModelType, language_model_name: str
) -> RunnableSerializable:
    database_retriever = create_database_retriever(embedding_database)
    llm = create_llm(language_model_type, language_model_name)

    question_answer_chain = generate_retrieval_chain(database_retriever, llm)

    return question_answer_chain


__all__ = [
    "create_embedding_database",
    "load_embedding_database",
    "load_source_documents",
    "prepare_question_answer_chain",
    "store_embedding_database",
]
