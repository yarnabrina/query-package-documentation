"""Define functionalities for top level modules."""

import logging
import pathlib
import shutil

import pydantic

from .dataset_generation import generate_json_dataset, generate_raw_datasets, store_json_dataset
from .information_retrieval import (
    PipelineType,
    RetrievalType,
    TransformerType,
    configure_language_model,
    create_embedding_database,
    load_embedding_database,
    load_source_documents,
    prepare_question_answer_chain,
    run_question_answer_chain,
    store_embedding_database,
)
from .utils_top_level import Response

LOGGER = logging.getLogger(__name__)


@pydantic.validate_call(validate_return=True)
def create_dataset(package_name: str, dataset_file: pathlib.Path, force: bool) -> pathlib.Path:
    """Generate JSON dataset for querying a package documentation.

    Parameters
    ----------
    package_name : str
        name of the root package to import with
    dataset_file : pathlib.Path
        path to store JSON dataset
    force : bool, optional
        override if ``dataset_file`` already exists

    Returns
    -------
    pathlib.Path
        absolute path storing JSON dataset

    Raises
    ------
    FileExistsError
        if ``dataset_file`` already exists and overriding is not allowed
    """
    if dataset_file.exists() and not force:
        LOGGER.error(f"{dataset_file=} refers to an existing file but {force=}")

        raise FileExistsError("Dataset exists already, aborting.")

    if dataset_file.exists():
        dataset_file.unlink()
        LOGGER.warning("Deleted existed dataset.")

    raw_datasets = generate_raw_datasets(package_name)
    json_dataset = generate_json_dataset(raw_datasets)

    store_json_dataset(json_dataset, dataset_file)

    return dataset_file.resolve()


@pydantic.validate_call(validate_return=True)
def create_database(
    dataset_file: pathlib.Path, embedding_model: str, database_directory: pathlib.Path, force: bool
) -> pathlib.Path:
    """Generate embedding database for querying a package documentation.

    Parameters
    ----------
    dataset_file : pathlib.Path
        path storing JSON dataset
    embedding_model : str
        name of Sentence Transformers model from Hugging Face
    database_directory : pathlib.Path
        path to directory for storing vector store
    force : bool
        override if ``database_directory`` already exists

    Returns
    -------
    pathlib.Path
        absolute path to directory storing vector store

    Raises
    ------
    FileExistsError
        if ``database_directory`` already exists and overriding is not allowed
    FileNotFoundError
        if ``dataset_file`` does not exist
    """
    if database_directory.exists() and not force:
        LOGGER.error(f"{database_directory=} refers to an existing directory but {force=}")

        raise FileExistsError("Database exists already, aborting.")

    if database_directory.exists():
        shutil.rmtree(database_directory)
        LOGGER.warning("Deleted existed database.")

    if not dataset_file.exists():
        LOGGER.error(f"{dataset_file=} refers to a non-existing file")

        raise FileNotFoundError("Dataset file is missing, aborting.")

    source_documents = load_source_documents(dataset_file)
    embedding_database = create_embedding_database(
        embedding_model, database_directory, source_documents
    )

    store_embedding_database(embedding_database)

    return database_directory.resolve()


@pydantic.validate_call(validate_return=True)
def get_response(  # noqa: PLR0913
    question: str,
    embedding_model: str,
    database_directory: pathlib.Path,
    search_type: RetrievalType,
    number_of_documents: int,
    initial_number_of_documents: int,
    diversity_level: float,
    language_model_type: TransformerType,
    standard_pipeline_type: PipelineType,
    standard_model_name: str,
    quantised_model_name: str,
    quantised_model_file: str,
    quantised_model_type: str,
) -> Response:
    """Get answer from large language model.

    Parameters
    ----------
    question : str
        query from user
    embedding_model : str
        name of Sentence Transformers model from Hugging Face
    database_directory : pathlib.Path
        path to load vector store from
    search_type : RetrievalType
        kind of retrieval algorithm for searching vector store
    number_of_documents : int
        number of documents to retrieve
    initial_number_of_documents : int
        initial number of documents to consider
    diversity_level : float
        similarity between retrieved documents
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
    Response
        answer from large language model with additional captured details

    Raises
    ------
    FileNotFoundError
        if ``database_directory`` does not exist
    """
    if not database_directory.exists():
        LOGGER.error(f"{database_directory=} refers to a non-existing directory")

        raise FileNotFoundError("Database directory is missing, aborting.")

    embedding_database = load_embedding_database(embedding_model, database_directory)
    language_model = configure_language_model(
        language_model_type,
        standard_pipeline_type,
        standard_model_name,
        quantised_model_name,
        quantised_model_file,
        quantised_model_type,
    )
    question_answer_chain = prepare_question_answer_chain(
        embedding_database,
        search_type,
        number_of_documents,
        initial_number_of_documents,
        diversity_level,
        language_model,
    )

    answer, callback = run_question_answer_chain(question_answer_chain, question)

    return Response.model_validate(
        {
            "query": answer["query"],
            "answer": answer["result"],
            "source_documents": [
                source_document.page_content for source_document in answer["source_documents"]
            ],
            "used_prompt": callback.effective_prompt,
            "llm_duration": callback.effective_duration,
        }
    )


__all__ = ["create_database", "create_dataset", "get_response"]
