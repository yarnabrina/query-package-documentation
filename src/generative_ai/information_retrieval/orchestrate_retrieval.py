import pathlib

import pydantic
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSerializable

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
    ValidatedChroma,
)


def load_source_documents(file_path: pathlib.Path) -> list[Document]:
    raw_documents = load_json_documents(file_path)
    partitioned_documents = partition_documents(raw_documents)

    return partitioned_documents


def create_embedding_database(
    embedding_model: str, directory_path: pathlib.Path, source_documents: list[Document]
) -> ValidatedChroma:
    document_embedder = create_document_embedder(embedding_model)

    vector_store = create_vector_store(document_embedder, directory_path)
    vector_store.add_documents(source_documents)

    return vector_store


def store_embedding_database(vector_store: ValidatedChroma) -> None:
    vector_store.persist()


def load_embedding_database(embedding_model: str, directory_path: pathlib.Path) -> ValidatedChroma:
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
    embedding_database: ValidatedChroma,
    search_type: RetrievalType,
    number_of_documents: int,
    initial_number_of_documents: int,
    diversity_level: float,
    language_model: LanguageModel,
) -> RunnableSerializable:
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
    question_answer_chain: RunnableSerializable, question: str
) -> tuple[dict, CaptureDetailsCallback]:
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
