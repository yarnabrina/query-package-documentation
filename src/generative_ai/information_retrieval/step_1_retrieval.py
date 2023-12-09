import pathlib

from langchain.docstore.document import Document
from langchain.document_loaders import JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma


def load_json_documents(file_path: pathlib.Path) -> list[Document]:
    json_loader = JSONLoader(file_path, ".documents[].retrieval_context")
    raw_documents = json_loader.load()

    return raw_documents


def partition_documents(raw_documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    partitioned_documents = text_splitter.split_documents(raw_documents)

    return partitioned_documents


def create_document_embedder(embedding_model: str) -> HuggingFaceEmbeddings:
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)

    return embedder


def create_vector_store(embedder: HuggingFaceEmbeddings, directory_path: pathlib.Path) -> Chroma:
    vector_store = Chroma(embedding_function=embedder, persist_directory=str(directory_path))

    return vector_store


__all__ = [
    "create_document_embedder",
    "create_vector_store",
    "load_json_documents",
    "partition_documents",
]
