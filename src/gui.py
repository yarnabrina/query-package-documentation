import gc
import pathlib

import gradio

from generative_ai.information_retrieval import PipelineType, RetrievalType, TransformerType
from generative_ai.top_level import create_database, create_dataset, get_response


def generate_dataset(
    package_name: str, dataset_file: pathlib.Path, force: bool = False
) -> pathlib.Path:
    try:
        dataset_path = create_dataset(package_name, dataset_file, force)
    except FileExistsError as error:
        raise gradio.Error(message=str(error)) from error
    else:
        gradio.Info("Dataset generation complete.")

        return dataset_path.resolve()


def generate_database(
    dataset_file: pathlib.Path, embedding_model: str, database_directory: pathlib.Path, force: bool
) -> pathlib.Path:
    try:
        database_path = create_database(dataset_file, embedding_model, database_directory, force)
    except (FileExistsError, FileNotFoundError) as error:
        raise gradio.Error(message=str(error)) from error
    else:
        gradio.Info("Database generation complete.")

        return database_path.resolve()


def answer_query(  # noqa: PLR0913
    query: str,
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
) -> tuple[str, list[str], str, float]:
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
        raise gradio.Error(message=str(error)) from error
    else:
        return (
            response.answer,
            response.source_documents,
            response.used_prompt,
            response.llm_duration,
        )
    finally:
        _ = gc.collect()


def step1_tab_flow() -> None:
    with gradio.Group():
        package_name_step1_input = gradio.Textbox(label="name to import package")
        dataset_file_step1_input = gradio.Textbox(
            value="json_documents.json", label="file where generated dataset needs to be stored"
        )

    force_step1_input = gradio.Checkbox(value=False, label="override existing dataset, if any")

    step1_button = gradio.Button(value="Generate Dataset")
    dataset_path_step1_output = gradio.Textbox(
        label="path to file storing dataset", show_copy_button=True
    )

    step1_button.click(
        generate_dataset,
        inputs=[package_name_step1_input, dataset_file_step1_input, force_step1_input],
        outputs=[dataset_path_step1_output],
    )


def step2_tab_flow() -> None:
    dataset_file_step2_input = gradio.Textbox(
        value="json_documents.json", label="path to file storing dataset"
    )

    with gradio.Group():
        embedding_model_step2_input = gradio.Textbox(
            value="sentence-transformers/all-MiniLM-L6-v2", label="embedding model to use"
        )
        database_directory_step2_input = gradio.Textbox(
            value="embeddings_database",
            label="directory where generated database needs to be stored",
        )

    force_step2_input = gradio.Checkbox(value=False, label="override existing database, if any")

    step2_button = gradio.Button(value="Generate Database")
    database_path_step2_output = gradio.Textbox(
        label="path to directory storing database", show_copy_button=True
    )

    step2_button.click(
        generate_database,
        inputs=[
            dataset_file_step2_input,
            embedding_model_step2_input,
            database_directory_step2_input,
            force_step2_input,
        ],
        outputs=[database_path_step2_output],
    )


def step3_tab_flow() -> None:
    query_step3_input = gradio.Textbox(label="user question")

    with gradio.Group():
        embedding_model_step3_input = gradio.Textbox(
            value="sentence-transformers/all-MiniLM-L6-v2", label="embedding model to use"
        )
        database_directory_step3_input = gradio.Textbox(
            value="embeddings_database", label="path to directory storing database"
        )

    with gradio.Accordion(label="Retrieval", open=False):
        search_type_step3_input = gradio.Radio(
            choices=[(element.name, element.value) for element in RetrievalType],
            value=RetrievalType.MMR.value,
            label="kind of retrieval",
        )
        number_of_documents_step3_input = gradio.Slider(
            minimum=1,
            maximum=10,
            value=3,
            step=1,
            label="number of documents to retrieve",
            randomize=False,
        )
        initial_number_of_documents_step3_input = gradio.Slider(
            minimum=2,
            maximum=30,
            value=5,
            step=3,
            label="initial number of documents to consider",
            randomize=False,
        )
        diversity_level_step3_input = gradio.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.1,
            label="similarity between retrieved documents",
            randomize=False,
        )

    with gradio.Accordion(label="Language Model", open=False):
        language_model_type_step3_input = gradio.Radio(
            choices=[(element.name, element.value) for element in TransformerType],
            value=TransformerType.STANDARD_TRANSFORMERS.value,
            label="kind of language model",
        )
        with gradio.Row():
            with gradio.Group():
                standard_pipeline_type_step3_input = gradio.Radio(
                    choices=[(element.name, element.value) for element in PipelineType],
                    value=PipelineType.TEXT2TEXT_GENERATION.value,
                    label="kind of Hugging Face pipeline",
                )
                standard_model_name_step3_input = gradio.Textbox(
                    value="google/flan-t5-large", label="name of Hugging Face model"
                )
            with gradio.Group():
                quantised_model_name_step3_input = gradio.Textbox(
                    value="TheBloke/zephyr-7B-beta-GGUF", label="name of Hugging Face model"
                )
                quantised_model_file_step3_input = gradio.Textbox(
                    value="zephyr-7b-beta.Q4_K_M.gguf", label="name of Hugging Face model file"
                )
                quantised_model_type_step3_input = gradio.Textbox(
                    value="mistral", label="type of Hugging Face model"
                )

    step3_button = gradio.Button(value="Get Response")

    with gradio.Row():
        with gradio.Group():
            llm_response_step3_output = gradio.Textbox(label="answer from language model")
            retrieved_context_step3_output = gradio.JSON(label="relevant documents")
            llm_duration_step3_output = gradio.Number(
                label="duration of language model in seconds"
            )

        with gradio.Group():
            llm_prompt_step3_output = gradio.Markdown(label="prompt used by language model")

    step3_button.click(
        answer_query,
        inputs=[
            query_step3_input,
            embedding_model_step3_input,
            database_directory_step3_input,
            search_type_step3_input,
            number_of_documents_step3_input,
            initial_number_of_documents_step3_input,
            diversity_level_step3_input,
            language_model_type_step3_input,
            standard_pipeline_type_step3_input,
            standard_model_name_step3_input,
            quantised_model_name_step3_input,
            quantised_model_file_step3_input,
            quantised_model_type_step3_input,
        ],
        outputs=[
            llm_response_step3_output,
            retrieved_context_step3_output,
            llm_prompt_step3_output,
            llm_duration_step3_output,
        ],
    )


def main() -> None:
    tab_title = "GUI for Generative AI aaplication"
    summary = """# Retrieval Augmented Generation from package docstrings .

## Dataset Generation

1. list all modules in the package (recursively from all sub-packages)
2. generate a set of documents based on package/module/object docstrings
3. documents are stored in a JSON dataset for retrieval (and tuning, optionally)

## Database Generation

1. read the retrieval dataset
2. generate embeddings for each document
3. store document embeddings in a vector database

## Response Generation

1. read the retrieval database
2. generate embeddings for user question
3. retrieve most similar documents from database
4. pass relevant documents to language model as context
5. generate answer using language model"""

    with gradio.Blocks(analytics_enabled=False, title=tab_title) as gui_application:
        _ = gradio.Markdown(value=summary, label="Description")

        with gradio.Tab(label="Step 1"):
            step1_tab_flow()

        with gradio.Tab(label="Step 2"):
            step2_tab_flow()

        with gradio.Tab(label="Step 3"):
            step3_tab_flow()

    gui_application.launch(share=False, show_error=True, show_api=False)


if __name__ == "__main__":
    main()
