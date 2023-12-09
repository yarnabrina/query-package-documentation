import pathlib
import shutil
import sys

import typer

CLI_APPLICATION = typer.Typer(name="CLI for Generative AI aaplication")


@CLI_APPLICATION.command()
def generate_dataset(
    package_name: str,
    dataset_file: pathlib.Path = pathlib.Path("json_documents.json"),
    force: bool = False,
) -> None:
    if dataset_file.exists() and not force:
        typer.echo("Dataset exists already, skipping. Use force if needed.")
        sys.exit(0)

    if dataset_file.exists():
        dataset_file.unlink()
        typer.echo("Deleted existed dataset.")

    from dataset_generation import generate_json_dataset, generate_raw_dataset, store_json_dataset

    raw_dataset = generate_raw_dataset(package_name)
    json_dataset = generate_json_dataset(raw_dataset)

    store_json_dataset(json_dataset, dataset_file)
    typer.echo("Dataset generation complete.")


@CLI_APPLICATION.command()
def generate_database(
    dataset_file: pathlib.Path = pathlib.Path("json_documents.json"),
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    database_directory: pathlib.Path = pathlib.Path("embeddings_database"),
    force: bool = False,
) -> None:
    if database_directory.exists() and not force:
        typer.echo("Database exists already, skipping. Use force if needed.")
        sys.exit(0)

    if database_directory.exists():
        shutil.rmtree(database_directory)
        typer.echo("Deleting existed database.")

    if not dataset_file.exists():
        typer.echo("Dataset file is missing, skipping. Use 'generate-dataset' first.")
        sys.exit(1)

    from information_retrieval import (
        create_embedding_database,
        load_source_documents,
        store_embedding_database,
    )

    source_documents = load_source_documents(dataset_file)
    embedding_database = create_embedding_database(
        embedding_model, database_directory, source_documents
    )

    store_embedding_database(embedding_database)
    typer.echo("Database generation complete.")


@CLI_APPLICATION.command()
def answer_query(
    query: str,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    database_directory: pathlib.Path = pathlib.Path("embeddings_database"),
    language_model: str = "google/flan-t5-large",
) -> None:
    if not database_directory.exists():
        typer.echo("Database directory is missing, skipping. Use 'generate-database' first.")
        sys.exit(1)

    from information_retrieval import load_embedding_database, prepare_question_answer_chain

    embedding_database = load_embedding_database(embedding_model, database_directory)
    question_answer_chain = prepare_question_answer_chain(embedding_database, language_model)

    answer = question_answer_chain.invoke(query)
    typer.echo(f"Answer: {answer['result']}.")


if __name__ == "__main__":
    CLI_APPLICATION()
