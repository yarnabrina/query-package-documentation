import logging
import pathlib
import shutil

import pydantic

from .dataset_generation import generate_json_dataset, generate_raw_datasets, store_json_dataset
from .information_retrieval import (
    LanguageModelType,
    create_embedding_database,
    load_embedding_database,
    load_source_documents,
    prepare_question_answer_chain,
    store_embedding_database,
)
from .utils_top_level import Response

LOGGER = logging.getLogger(__name__)


@pydantic.validate_call(validate_return=True)
def create_dataset(
    package_name: str, dataset_file: pathlib.Path, force: bool = False
) -> pathlib.Path:
    if dataset_file.exists() and not force:
        LOGGER.error(f"{dataset_file=} refers to an existing file but {force=}")

        raise FileExistsError("Dataset exists already, skipping. Use force if needed.")

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
    if database_directory.exists() and not force:
        LOGGER.error(f"{database_directory=} refers to an existing file but {force=}")

        raise FileExistsError("Dataset exists already, skipping. Use force if needed.")

    if database_directory.exists():
        shutil.rmtree(database_directory)
        LOGGER.warning("Deleted existed database.")

    if not dataset_file.exists():
        LOGGER.error(f"{dataset_file=} refers to a non-existing file")

        raise FileNotFoundError("Dataset file is missing, skipping. Use 'generate-dataset' first.")

    source_documents = load_source_documents(dataset_file)
    embedding_database = create_embedding_database(
        embedding_model, database_directory, source_documents
    )

    store_embedding_database(embedding_database)

    return database_directory.resolve()


@pydantic.validate_call(validate_return=True)
def get_response(
    question: str,
    embedding_model: str,
    database_directory: pathlib.Path,
    language_model_type: LanguageModelType,
    language_model_name: str,
) -> Response:
    if not database_directory.exists():
        LOGGER.error(f"{database_directory=} refers to a non-existing directory")

        raise FileNotFoundError(
            "Database directory is missing, skipping. Use 'generate-database' first."
        )

    embedding_database = load_embedding_database(embedding_model, database_directory)
    question_answer_chain = prepare_question_answer_chain(
        embedding_database, language_model_type, language_model_name
    )

    answer = question_answer_chain.invoke(question)

    return Response.model_validate(
        {
            "query": answer["query"],
            "answer": answer["result"],
            "source_documents": [
                source_document.page_content for source_document in answer["source_documents"]
            ],
        }
    )


__all__ = ["create_database", "create_dataset", "get_response"]
