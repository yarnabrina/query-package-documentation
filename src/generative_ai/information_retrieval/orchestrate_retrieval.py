"""Define functionalities to orchestrate information retrieval."""

import typing

import pydantic

from .step_1_retrieval import (
    create_document_embedder,
    create_vector_store,
    load_json_documents,
    partition_documents,
)
from .step_2_retrieval import create_database_retriever, create_llm, generate_retrieval_chain
from .step_3_retrieval import CaptureDetailsCallback
from .utils_retrieval import (
    LanguageModel,
    LanguageModelAdapter,
    PipelineType,
    RetrievalType,
    TransformerType,
)

if typing.TYPE_CHECKING:
    import pathlib

    from langchain.docstore.document import Document
    from langchain.schema.runnable import RunnableSerializable
    from langchain.vectorstores.chroma import Chroma


def load_source_documents(file_path: "pathlib.Path") -> list["Document"]:
    """Load and partition source documents.

    Parameters
    ----------
    file_path : pathlib.Path
        path storing JSON dataset

    Returns
    -------
    list[Document]
        partitioned source documents
    """
    raw_documents = load_json_documents(file_path)
    partitioned_documents = partition_documents(raw_documents)

    return partitioned_documents


def create_embedding_database(
    embedding_model: str, directory_path: "pathlib.Path", source_documents: list["Document"]
) -> "Chroma":
    """Prepare an embedding database.

    Parameters
    ----------
    embedding_model : str
        name of Sentence Transformers model from Hugging Face
    directory_path : pathlib.Path
        path to directory for storing vector store
    source_documents : list[Document]
        partitioned source documents

    Returns
    -------
    Chroma
        vector store
    """
    document_embedder = create_document_embedder(embedding_model)

    vector_store = create_vector_store(document_embedder, directory_path)
    vector_store.add_documents(source_documents)

    return vector_store


def store_embedding_database(vector_store: "Chroma") -> None:
    """Dump vector store to disk into configured directory.

    Parameters
    ----------
    vector_store : Chroma
        vector store
    """
    vector_store.persist()


def load_embedding_database(embedding_model: str, directory_path: "pathlib.Path") -> "Chroma":
    """Load vector store from disk from configured directory.

    Parameters
    ----------
    embedding_model : str
        name of Sentence Transformers model from Hugging Face
    directory_path : pathlib.Path
        path to load vector store from

    Returns
    -------
    Chroma
        vector store

    Notes
    -----
    * ``embedding_model`` must match the one originally used for database creation.
    """
    document_embedder = create_document_embedder(embedding_model)

    vector_store = create_vector_store(document_embedder, directory_path)

    return vector_store


@pydantic.validate_call(validate_return=True)
def configure_language_model(  # noqa: PLR0913
    language_model_type: TransformerType,
    standard_pipeline_type: PipelineType,
    standard_model_name: str,
    quantised_model_name: str,
    quantised_model_file: str,
    quantised_model_type: str,
) -> LanguageModel:
    """Prepare configurations to load language model.

    Parameters
    ----------
    language_model_type : TransformerType
        kind of language model
    standard_pipeline_type : PipelineType
        kind of Hugging Face pipeline
    standard_model_name : str
        name of ``transformers`` compatible Hugging Face model
    quantised_model_name : str
        name of ``ctransformers`` compatible Hugging Face model
    quantised_model_file : str
        named of quantised model file
    quantised_model_type : str
        type of quantised model

    Returns
    -------
    LanguageModel
        configurations of language model

    Raises
    ------
    ValueError
        if language model type is not supported
    """
    language_model: dict = {"language_model_type": language_model_type}

    match language_model_type:
        case TransformerType.STANDARD_TRANSFORMERS:
            language_model.update(
                {
                    "standard_pipeline_type": standard_pipeline_type,
                    "standard_model_name": standard_model_name,
                }
            )
        case TransformerType.QUANTISED_CTRANSFORMERS:
            language_model.update(
                {
                    "quantised_model_name": quantised_model_name,
                    "quantised_model_file": quantised_model_file,
                    "quantised_model_type": quantised_model_type,
                }
            )
        case _:
            raise ValueError("Unexpected language model type")

    return LanguageModelAdapter.validate_python(language_model)


def prepare_question_answer_chain(  # noqa: PLR0913
    embedding_database: "Chroma",
    search_type: RetrievalType,
    number_of_documents: int,
    initial_number_of_documents: int,
    diversity_level: float,
    language_model: LanguageModel,
) -> "RunnableSerializable":
    """Prepare a question answering pipeline.

    Parameters
    ----------
    embedding_database : Chroma
        vector store
    search_type : RetrievalType
        kind of retrieval algorithm for searching vector store
    number_of_documents : int
        number of documents to retrieve
    initial_number_of_documents : int
        initial number of documents to consider
    diversity_level : float
        similarity between retrieved documents
    language_model : LanguageModel
        configurations of language model

    Returns
    -------
    RunnableSerializable
        question answering pipeline
    """
    database_retriever = create_database_retriever(
        embedding_database,
        search_type,
        number_of_documents,
        initial_number_of_documents,
        diversity_level,
    )
    llm = create_llm(language_model)

    question_answer_chain = generate_retrieval_chain(database_retriever, llm)

    return question_answer_chain


def run_question_answer_chain(
    question_answer_chain: "RunnableSerializable", question: str
) -> tuple[dict, CaptureDetailsCallback]:
    """Run question answering pipeline for user input.

    Parameters
    ----------
    question_answer_chain : RunnableSerializable
        question answering pipeline
    question : str
        query from user

    Returns
    -------
    dict
        response from large language model
    CaptureDetailsCallback
        callback capturing details of particular run of question answering pipeline
    """
    details_callback = CaptureDetailsCallback()
    answer = question_answer_chain.invoke(question, config={"callbacks": [details_callback]})

    return answer, details_callback


__all__ = [
    "configure_language_model",
    "create_embedding_database",
    "load_embedding_database",
    "load_source_documents",
    "prepare_question_answer_chain",
    "run_question_answer_chain",
    "store_embedding_database",
]
