import pathlib

import gradio

from generative_ai.information_retrieval import LanguageModelType
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


def answer_query(
    query: str,
    embedding_model: str,
    database_directory: pathlib.Path,
    language_model_type: LanguageModelType,
    language_model_name: str,
) -> str:
    try:
        response = get_response(
            query, embedding_model, database_directory, language_model_type, language_model_name
        )
    except FileNotFoundError as error:
        raise gradio.Error(message=str(error)) from error
    else:
        return response.answer


def step1_tab_flow() -> None:
    with gradio.Row():
        package_name_step1_input = gradio.Textbox(label="name to import package")
        dataset_file_step1_input = gradio.Textbox(
            value="json_documents.json", label="file where generated dataset needs to be stored"
        )

    force_step1_checkbox = gradio.Checkbox(value=False, label="override existing dataset, if any")

    step1_button = gradio.Button(value="Generate Dataset")
    step1_output = gradio.Textbox(label="path to file storing dataset", show_copy_button=True)

    step1_button.click(
        generate_dataset,
        inputs=[package_name_step1_input, dataset_file_step1_input, force_step1_checkbox],
        outputs=[step1_output],
    )


def step2_tab_flow() -> None:
    with gradio.Row():
        dataset_file_step2_input = gradio.File(
            file_types=[".json"], label="path to file storing dataset"
        )
        embedding_model_step2_input = gradio.Textbox(
            value="sentence-transformers/all-MiniLM-L6-v2", label="embedding model to use"
        )
        database_directory_step2_input = gradio.Textbox(
            value="embeddings_database",
            label="directory where generated database needs to be stored",
        )

    force_step2_checkbox = gradio.Checkbox(value=False, label="override existing database, if any")

    step2_button = gradio.Button(value="Generate Database")
    step2_output = gradio.Textbox(
        label="path to directory storing database", show_copy_button=True
    )

    step2_button.click(
        generate_database,
        inputs=[
            dataset_file_step2_input,
            embedding_model_step2_input,
            database_directory_step2_input,
            force_step2_checkbox,
        ],
        outputs=[step2_output],
    )


def step3_tab_flow() -> None:
    query_step3_input = gradio.Textbox(label="user question")

    with gradio.Row():
        embedding_model_step3_input = gradio.Textbox(
            value="sentence-transformers/all-MiniLM-L6-v2", label="embedding model to use"
        )
        database_directory_step3_input = gradio.Textbox(label="path to directory storing database")

    with gradio.Row():
        language_model_type_step3_input = gradio.Radio(
            choices=[(element.name, element.value) for element in LanguageModelType],
            value=LanguageModelType.HUGGINGFACE_STANDARD.value,
            label="kind of language model",
        )
        language_model_name_step3_input = gradio.Textbox(
            value="google/flan-t5-large", label="name of Hugging Face model"
        )

    step3_button = gradio.Button(value="Get Response")
    step3_output = gradio.Textbox(label="answer from language model")

    step3_button.click(
        answer_query,
        inputs=[
            query_step3_input,
            embedding_model_step3_input,
            database_directory_step3_input,
            language_model_type_step3_input,
            language_model_name_step3_input,
        ],
        outputs=[step3_output],
    )


def main() -> None:
    with gradio.Blocks(title="GUI for Generative AI aaplication") as gui_application:
        with gradio.Tab("Step 1"):
            step1_tab_flow()

        with gradio.Tab("Step 2"):
            step2_tab_flow()

        with gradio.Tab("Step 3"):
            step3_tab_flow()

    gui_application.launch(share=False, show_error=True, show_api=False)


if __name__ == "__main__":
    main()
