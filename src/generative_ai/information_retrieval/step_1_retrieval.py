"""Define functionalities to store document embeddings."""

import typing

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

if typing.TYPE_CHECKING:
    import pathlib

    from langchain.docstore.document import Document


def load_json_documents(file_path: "pathlib.Path") -> list["Document"]:
    """Load retrieval documents from a JSON file.

    Parameters
    ----------
    file_path : pathlib.Path
        path to JSON file

    Returns
    -------
    list[Document]
        retrieval documents
    """
    json_loader = JSONLoader(file_path, ".retrieval_documents[]")
    raw_documents = json_loader.load()

    return raw_documents


def partition_documents(raw_documents: list["Document"]) -> list["Document"]:
    """Partition retrieval documents into chunks.

    Parameters
    ----------
    raw_documents : list[Document]
        retrieval documents

    Returns
    -------
    list[Document]
        chunks of retrieval documents

    Notes
    -----
    * Chunk length will be at most 512 tokens.
    * Different chunks from same document will overlap by 64 tokens.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    partitioned_documents = text_splitter.split_documents(raw_documents)

    return partitioned_documents


def create_document_embedder(embedding_model: str) -> HuggingFaceEmbeddings:
    """Prepare a Sentence Transformers model for document embedding.

    Parameters
    ----------
    embedding_model : str
        name of Sentence Transformers model from Hugging Face

    Returns
    -------
    HuggingFaceEmbeddings
        document embedder
    """
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)

    return embedder


def create_vector_store(embedder: HuggingFaceEmbeddings, directory_path: "pathlib.Path") -> Chroma:
    """Initialise a Chroma vector store.

    Parameters
    ----------
    embedder : HuggingFaceEmbeddings
        document embedder
    directory_path : pathlib.Path
        path to directory for storing vector store

    Returns
    -------
    Chroma
        vector store
    """
    vector_store = Chroma(embedding_function=embedder, persist_directory=str(directory_path))

    return vector_store


__all__ = [
    "create_document_embedder",
    "create_vector_store",
    "load_json_documents",
    "partition_documents",
]
