"""Define command line interface using Typer."""

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
    """Create JSON dataset for querying a package documentation.

    Parameters
    ----------
    package_name : str
        name of the root package to import with
    dataset_file : pathlib.Path, optional
        path to store JSON dataset, by default pathlib.Path("json_documents.json")
    force : bool, optional
        override if ``dataset_file`` already exists, by default False
    """
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
    """Generate embedding database for querying a package documentation.

    Parameters
    ----------
    dataset_file : pathlib.Path, optional
        path storing JSON dataset, by default pathlib.Path("json_documents.json")
    embedding_model : str, optional
        name of Sentence Transformers model, by default "sentence-transformers/all-MiniLM-L6-v2"
    database_directory : pathlib.Path, optional
        path to directory for storing vector store, by default pathlib.Path("embeddings_database")
    force : bool, optional
        override if ``database_directory`` already exists, by default False
    """
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
    search_type: RetrievalType = RetrievalType.SIMILARITY,
    number_of_documents: int = 5,
    initial_number_of_documents: int = 10,
    diversity_level: float = 0.5,
    language_model_type: TransformerType = TransformerType.STANDARD_TRANSFORMERS,
    standard_pipeline_type: PipelineType = PipelineType.TEXT2TEXT_GENERATION,
    standard_model_name: str = "google/flan-t5-large",
    quantised_model_name: str = "TheBloke/zephyr-7B-beta-GGUF",
    quantised_model_file: str = "zephyr-7b-beta.Q4_K_M.gguf",
    quantised_model_type: str = "mistral",
) -> None:
    """Get response from large language model.

    Parameters
    ----------
    query : str
        question from user
    embedding_model : str, optional
        name of Sentence Transformers model, by default "sentence-transformers/all-MiniLM-L6-v2"
    database_directory : pathlib.Path, optional
        path to directory for storing vector store, by default pathlib.Path("embeddings_database")
    search_type : RetrievalType, optional
        kind of retrieval algorithm for searching vector store, by default RetrievalType.SIMILARITY
    number_of_documents : int, optional
        number of documents to retrieve, by default 5
    initial_number_of_documents : int, optional
        initial number of documents to consider, by default 10
    diversity_level : float, optional
        similarity between retrieved documents, by default 0.5
    language_model_type : TransformerType, optional
        kind of language model, by default TransformerType.STANDARD_TRANSFORMERS
    standard_pipeline_type : PipelineType, optional
        kind of Hugging Face pipeline, by default PipelineType.TEXT2TEXT_GENERATION
    standard_model_name : str, optional
        name of ``transformers`` compatible model, by default "google/flan-t5-large"
    quantised_model_name : str, optional
        name of ``ctransformers`` compatible model, by default "TheBloke/zephyr-7B-beta-GGUF"
    quantised_model_file : str, optional
        named of quantised model file, by default "zephyr-7b-beta.Q4_K_M.gguf"
    quantised_model_type : str, optional
        type of quantised model, by default "mistral"
    """
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
