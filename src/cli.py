import pathlib
import sys

import typer

from generative_ai.information_retrieval import LanguageModelType
from generative_ai.top_level import create_database, create_dataset, get_response

CLI_APPLICATION = typer.Typer(name="CLI for Generative AI aaplication")


@CLI_APPLICATION.command()
def generate_dataset(
    package_name: str,
    dataset_file: pathlib.Path = pathlib.Path("json_documents.json"),
    force: bool = False,
) -> None:
    try:
        dataset_path = create_dataset(package_name, dataset_file, force)
    except FileExistsError as error:
        typer.echo(message=str(error), err=True)
        sys.exit(1)
    else:
        typer.echo(f"Dataset generation complete: '{dataset_path}'.")


@CLI_APPLICATION.command()
def generate_database(
    dataset_file: pathlib.Path = pathlib.Path("json_documents.json"),
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    database_directory: pathlib.Path = pathlib.Path("embeddings_database"),
    force: bool = False,
) -> None:
    try:
        database_path = create_database(dataset_file, embedding_model, database_directory, force)
    except (FileExistsError, FileNotFoundError) as error:
        typer.echo(message=str(error), err=True)
        sys.exit(1)
    else:
        typer.echo(f"Database generation complete: '{database_path}'.")


@CLI_APPLICATION.command()
def answer_query(
    query: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    database_directory: pathlib.Path = pathlib.Path("embeddings_database"),
    language_model_type: LanguageModelType = LanguageModelType.HUGGINGFACE_STANDARD,
    language_model_name: str = "google/flan-t5-large",
) -> None:
    try:
        response = get_response(
            query, embedding_model, database_directory, language_model_type, language_model_name
        )
    except FileNotFoundError as error:
        typer.echo(message=str(error), err=True)
        sys.exit(1)
    else:
        typer.echo(f"Query: {response.query}")
        typer.echo(f"Answer: {response.answer}")

        for counter, source_document in enumerate(response.source_documents):
            typer.echo(f"Source {counter + 1}: {source_document}")


if __name__ == "__main__":
    CLI_APPLICATION()
