import pathlib
import sys

import typer

from generative_ai.information_retrieval import PipelineType, RetrievalType, TransformerType
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
def answer_query(  # noqa: PLR0913
    query: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    database_directory: pathlib.Path = pathlib.Path("embeddings_database"),
    search_type: RetrievalType = RetrievalType.MMR,
    number_of_documents: int = 3,
    initial_number_of_documents: int = 5,
    diversity_level: float = 0.5,
    language_model_type: TransformerType = TransformerType.STANDARD_TRANSFORMERS,
    standard_pipeline_type: PipelineType = PipelineType.TEXT2TEXT_GENERATION,
    standard_model_name: str = "google/flan-t5-large",
    quantised_model_name: str = "TheBloke/zephyr-7B-beta-GGUF",
    quantised_model_file: str = "zephyr-7b-beta.Q4_K_M.gguf",
    quantised_model_type: str = "mistral",
) -> None:
    try:
        response = get_response(
            query,
            embedding_model,
            database_directory,
            search_type,
            number_of_documents,
            initial_number_of_documents,
            diversity_level,
            language_model_type,
            standard_pipeline_type,
            standard_model_name,
            quantised_model_name,
            quantised_model_file,
            quantised_model_type,
        )
    except FileNotFoundError as error:
        typer.echo(message=str(error), err=True)
        sys.exit(1)
    else:
        typer.echo(f"Query: {response.query}")
        typer.echo(f"Answer: {response.answer}")
        typer.echo(f"Duration: {response.llm_duration:.2f} seconds")

        for counter, source_document in enumerate(response.source_documents):
            typer.echo(f"Source {counter + 1}: {source_document}")


if __name__ == "__main__":
    CLI_APPLICATION()
