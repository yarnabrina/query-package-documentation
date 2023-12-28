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
) -> tuple[str, list[str]]:
    try:
        response = get_response(
            query, embedding_model, database_directory, language_model_type, language_model_name
        )
    except FileNotFoundError as error:
        raise gradio.Error(message=str(error)) from error
    else:
        return response.answer, response.source_documents


def step1_tab_flow() -> gradio.Textbox:
    with gradio.Group():
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

    return step1_output


def step2_tab_flow(dataset_path: gradio.Textbox) -> tuple[gradio.Textbox, gradio.Textbox]:
    if dataset_path.value is None:
        dataset_file_step2_input = dataset_path
    else:
        dataset_file_step2_input = gradio.Textbox(label="path to file storing dataset")

    with gradio.Group():
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

    return step2_output, embedding_model_step2_input


def step3_tab_flow(database_path: gradio.Textbox, embedding_model: gradio.Textbox) -> None:
    query_step3_input = gradio.Textbox(label="user question")

    with gradio.Group():
        if database_path.value is not None:
            database_directory_step3_input = database_path
        else:
            database_directory_step3_input = gradio.Textbox(
                label="path to directory storing database"
            )

        if embedding_model.value is not None:
            embedding_model_step3_input = embedding_model
        else:
            embedding_model_step3_input = gradio.Textbox(
                value="sentence-transformers/all-MiniLM-L6-v2", label="embedding model to use"
            )

    with gradio.Accordion(label="Language Model", open=False):
        language_model_type_step3_input = gradio.Radio(
            choices=[(element.name, element.value) for element in LanguageModelType],
            value=LanguageModelType.HUGGINGFACE_STANDARD.value,
            label="kind of language model",
        )
        language_model_name_step3_input = gradio.Textbox(
            value="google/flan-t5-large", label="name of Hugging Face model"
        )

    step3_button = gradio.Button(value="Get Response")

    with gradio.Group():
        step3_output = gradio.Textbox(label="answer from language model")
        step3_additional_outputs = gradio.JSON(label="relevant documents")

    step3_button.click(
        answer_query,
        inputs=[
            query_step3_input,
            embedding_model_step3_input,
            database_directory_step3_input,
            language_model_type_step3_input,
            language_model_name_step3_input,
        ],
        outputs=[step3_output, step3_additional_outputs],
    )


def main() -> None:
    gui_application_title = "GUI for Generative AI aaplication"
    gui_application_description = """# Retrieval Augmented Generation from package docstrings .

## Dataset Generation

1. list all modules in the package (recursively from all sub-packages)
2. generate a set of documents based on package/module/object docstrings
3. documents are stored in a JSON dataset for retrieval (and tuning, optionally)

## Database Generation

1. read the retrieval dataset
2. generate embeddings for each document
3. store dpcument embeddings in a vector database

## Response Generation

1. read the retrieval database
2. generate embeddings for user question
3. retrieve most similar documents from database
4. pass relevant documents to language model as context
5. generate answer using language model"""

    with gradio.Blocks(title=gui_application_title) as gui_application:
        _ = gradio.Markdown(value=gui_application_description, label="Description")

        with gradio.Tabs() as steps:
            with gradio.Tab(label="Step 1", id=1):
                dataset_path = step1_tab_flow()
                dataset_path.select(lambda: gradio.Tabs(selected=2), outputs=[steps])

            with gradio.Tab(label="Step 2", id=2):
                database_path, embedding_model = step2_tab_flow(dataset_path)
                database_path.select(lambda: gradio.Tabs(selected=3), outputs=[steps])

            with gradio.Tab(label="Step 3", id=3):
                step3_tab_flow(database_path, embedding_model)

    gui_application.launch(share=False, show_error=True, show_api=False)


if __name__ == "__main__":
    main()
