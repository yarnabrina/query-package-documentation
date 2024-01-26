"""Define graphical user interface using Gradio."""

import gc
import pathlib

import gradio

from generative_ai.information_retrieval import PipelineType, RetrievalType, TransformerType
from generative_ai.top_level import create_database, create_dataset, get_response

TITLE = "GUI for Generative AI application"
SUMMARY = """# Retrieval Augmented Generation from package docstrings .

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


def validate_dataset(dataset_file: pathlib.Path) -> pathlib.Path:
    """Check if dataset exists already.

    Parameters
    ----------
    dataset_file : pathlib.Path
        path to directory for storing vector store

    Returns
    -------
    pathlib.Path
        absolute path to file storing JSON dataset

    Raises
    ------
    gradio.Error
        if ``dataset_file`` does not exist
    """
    dataset_file = pathlib.Path(dataset_file)

    if dataset_file.exists():
        return dataset_file.resolve()

    raise gradio.Error(message=f"Dataset does not exist at {dataset_file}.")


def validate_database(database_directory: pathlib.Path) -> pathlib.Path:
    """Check if dataset exists already.

    Parameters
    ----------
    database_directory : pathlib.Path
        path to directory storing vector store

    Returns
    -------
    pathlib.Path
        absolute path to directory storing vector store

    Raises
    ------
    gradio.Error
        if ``database_directory`` does not exist
    """
    database_directory = pathlib.Path(database_directory)

    if database_directory.exists():
        return database_directory.resolve()

    raise gradio.Error(message=f"Database does not exist at {database_directory}.")


def switch_tab() -> tuple[gradio.Tab, gradio.Tab]:
    """Modify interactive state of tabs.

    Returns
    -------
    gradio.Tab
        updated current tab as non-interactive
    gradio.Tab
        updated new (previous or next) tab as interactive
    """
    updated_source_tab = gradio.Tab(interactive=False)
    updated_destnation_tab = gradio.Tab(interactive=True)

    return updated_source_tab, updated_destnation_tab


def activate_button() -> gradio.Button:
    """Make button interactive.

    Returns
    -------
    gradio.Button
        updated button as interactive
    """
    updated_button = gradio.Button(interactive=True)

    return updated_button


def update_textbox_value(text: str) -> gradio.Textbox:
    """Update textbox with value from another textbox.

    Parameters
    ----------
    text : str
        value to update textbox with

    Returns
    -------
    gradio.Textbox
        updated textbox with value from ``text``
    """
    updated_textbox = gradio.Textbox(value=text)

    return updated_textbox


def generate_dataset(package_name: str, dataset_file: pathlib.Path, force: bool) -> pathlib.Path:
    """Create JSON dataset for querying a package documentation.

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
    gradio.Error
        if ``dataset_file`` already exists and overriding is not allowed
    """
    try:
        dataset_path = create_dataset(package_name, dataset_file, force)
    except FileExistsError as error:
        raise gradio.Error(message=str(error)) from error

    gradio.Info("Dataset generation complete.")

    return dataset_path.resolve()


def generate_database(
    dataset_file: pathlib.Path, embedding_model: str, database_directory: pathlib.Path, force: bool
) -> tuple[str, pathlib.Path]:
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
    str
        name of Sentence Transformers model from Hugging Face used to create vector store
    pathlib.Path
        absolute path to directory storing vector store

    Raises
    ------
    gradio.Error
        if ``database_directory`` already exists and overriding is not allowed
        or if ``dataset_file`` does not exist
    """
    try:
        database_path = create_database(dataset_file, embedding_model, database_directory, force)
    except (FileExistsError, FileNotFoundError) as error:
        raise gradio.Error(message=str(error)) from error

    gradio.Info("Database generation complete.")

    return embedding_model, database_path.resolve()


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
    """Get response from large language model.

    Parameters
    ----------
    query : str
        question from user
    embedding_model : str
        name of Sentence Transformers model used for vector store
    database_directory : pathlib.Path
        path to directory storing vector store
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
    str
        response from large language model
    list[str]
        list of source documents retrieved from database
    str
        exact prompt passed to large language model
    float
        time taken (in seconds) for large language model to generate response

    Raises
    ------
    gradio.Error
        if ``database_directory`` does not exist
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


def summary_tab_flow() -> None:
    """Describe different functionalities of the application."""
    _ = gradio.Markdown(
        value=SUMMARY, label="Description", show_label=True, elem_id="application summary"
    )


def step1_tab_flow() -> gradio.Textbox:
    """Orchestrate flow of first step to generate retieval and tuning documents.

    Returns
    -------
    gradio.Textbox
        absolute path to file storing JSON dataset
    """
    with gradio.Group(elem_id="step 1 mandatory inputs"):
        package_name_step1_input = gradio.Textbox(
            label="name to import package", show_label=True, elem_id="package name step 1 input"
        )

    with gradio.Group(elem_id="step 1 optional inputs"):
        dataset_file_step1_input = gradio.Textbox(
            value="json_documents.json",
            label="file where generated dataset needs to be stored",
            show_label=True,
            elem_id="dataset file step 1 input",
        )

    force_step1_input = gradio.Checkbox(
        value=False,
        label="override existing dataset, if any",
        show_label=True,
        elem_id="force step 1 input",
    )

    with gradio.Row(elem_id="step 1"):
        skip_step1_button = gradio.Button(
            value="Validate Dataset", variant="secondary", elem_id="skip step 1 button"
        )
        run_step1_button = gradio.Button(
            value="Generate Dataset", variant="primary", elem_id="run step 1 button"
        )

    with gradio.Group(elem_id="step 1 output"):
        dataset_path_step1_output = gradio.Textbox(
            label="path to file storing dataset",
            show_label=True,
            interactive=False,
            elem_id="dataset path step 1 output",
            show_copy_button=True,
        )

    _ = skip_step1_button.click(
        validate_dataset, inputs=[dataset_file_step1_input], outputs=[dataset_path_step1_output]
    )
    _ = run_step1_button.click(
        generate_dataset,
        inputs=[package_name_step1_input, dataset_file_step1_input, force_step1_input],
        outputs=[dataset_path_step1_output],
    )

    return dataset_path_step1_output


def step2_tab_flow() -> tuple[gradio.Textbox, gradio.Textbox, gradio.Textbox]:
    """Orchestrate flow of second step to generate vector embeddings for retrieval.

    Returns
    -------
    gradio.Textbox
        absolute path to file storing JSON dataset
    gradio.Textbox
        Sentence Transformers model used to create vector store
    gradio.Textbox
        absolute path to directory storing vector store
    """
    with gradio.Group(elem_id="step 2 mandatory inputs"):
        dataset_path_step2_input = gradio.Textbox(
            label="path to file storing dataset",
            show_label=True,
            elem_id="dataset file step 2 input",
        )

    with gradio.Group(elem_id="step 2 optional inputs"):
        embedding_model_step2_input = gradio.Textbox(
            value="sentence-transformers/all-MiniLM-L6-v2",
            label="embedding model to use",
            show_label=True,
            elem_id="embedding model step 2 input",
        )
        database_directory_step2_input = gradio.Textbox(
            value="embeddings_database",
            label="directory where generated database needs to be stored",
            show_label=True,
            elem_id="database directory step 2 input",
        )

    force_step2_input = gradio.Checkbox(
        value=False,
        label="override existing database, if any",
        show_label=True,
        elem_id="force step 2 input",
    )

    with gradio.Row(elem_id="step 2"):
        skip_step2_button = gradio.Button(
            value="Validate Database", variant="secondary", elem_id="skip step 2 button"
        )
        run_step2_button = gradio.Button(
            value="Generate Database", variant="primary", elem_id="run step 2 button"
        )

    with gradio.Group(elem_id="step 2 outputs"):
        database_path_step2_output = gradio.Textbox(
            label="path to directory storing database",
            show_label=True,
            interactive=False,
            elem_id="database path step 2 output",
            show_copy_button=True,
        )
        embedding_model_step2_output = gradio.Textbox(
            label="used embedding model",
            show_label=True,
            interactive=False,
            elem_id="embedding model step 2 output",
        )

    _ = skip_step2_button.click(
        validate_dataset,
        inputs=[database_directory_step2_input],
        outputs=[database_path_step2_output],
    ).success(
        update_textbox_value,
        inputs=[embedding_model_step2_input],
        outputs=[embedding_model_step2_output],
    )
    _ = run_step2_button.click(
        generate_database,
        inputs=[
            dataset_path_step2_input,
            embedding_model_step2_input,
            database_directory_step2_input,
            force_step2_input,
        ],
        outputs=[embedding_model_step2_output, database_path_step2_output],
    )

    return dataset_path_step2_input, embedding_model_step2_output, database_path_step2_output


def step3_tab_flow() -> tuple[gradio.Textbox, gradio.Textbox]:
    """Orchestrate flow of third step to generate response from large language model.

    Returns
    -------
    gradio.Textbox
        absolute path to directory storing vector store
    gradio.Textbox
        Sentence Transformers model used to create vector store
    """
    query_step3_input = gradio.Textbox(
        label="user question", show_label=True, elem_id="query step 3 input"
    )

    with gradio.Group(elem_id="step 3 optional embedding inputs"):
        database_path_step3_input = gradio.Textbox(
            value="embeddings_database",
            label="path to directory storing database",
            show_label=True,
            elem_id="database directory step 3 input",
        )
        embedding_model_step3_input = gradio.Textbox(
            value="sentence-transformers/all-MiniLM-L6-v2",
            label="embedding model used in vector database",
            show_label=True,
            elem_id="embedding model step 3 input",
        )

    with gradio.Accordion(
        label="Retrieval", open=False, elem_id="step 3 optional retrieval inputs"
    ):
        search_type_step3_input = gradio.Radio(
            choices=[(element.name, element.value) for element in RetrievalType],
            value=RetrievalType.SIMILARITY.value,
            label="kind of retrieval",
            show_label=True,
            elem_id="search type step 3 input",
        )
        number_of_documents_step3_input = gradio.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="number of documents to retrieve",
            show_label=True,
            elem_id="number of documents step 3 input",
            randomize=False,
        )
        initial_number_of_documents_step3_input = gradio.Slider(
            minimum=2,
            maximum=30,
            value=10,
            step=3,
            label="initial number of documents to consider",
            show_label=True,
            elem_id="initial number of documents step 3 input",
            randomize=False,
        )
        diversity_level_step3_input = gradio.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.1,
            label="similarity between retrieved documents",
            show_label=True,
            elem_id="diversity level step 3 input",
            randomize=False,
        )

    with gradio.Accordion(
        label="Language Model", open=False, elem_id="step 3 optional language model inputs"
    ):
        language_model_type_step3_input = gradio.Radio(
            choices=[(element.name, element.value) for element in TransformerType],
            value=TransformerType.STANDARD_TRANSFORMERS.value,
            label="kind of language model",
            show_label=True,
            elem_id="language model type step 3 input",
        )
        with gradio.Row():
            with gradio.Group():
                standard_pipeline_type_step3_input = gradio.Radio(
                    choices=[(element.name, element.value) for element in PipelineType],
                    value=PipelineType.TEXT2TEXT_GENERATION.value,
                    label="kind of Hugging Face pipeline",
                    show_label=True,
                    elem_id="standard pipeline type step 3 input",
                )
                standard_model_name_step3_input = gradio.Textbox(
                    value="google/flan-t5-large",
                    label="name of Hugging Face model",
                    show_label=True,
                    elem_id="standard model name step 3 input",
                )
            with gradio.Group():
                quantised_model_name_step3_input = gradio.Textbox(
                    value="TheBloke/zephyr-7B-beta-GGUF",
                    label="name of Hugging Face model",
                    show_label=True,
                    elem_id="quantised model name step 3 input",
                )
                quantised_model_file_step3_input = gradio.Textbox(
                    value="zephyr-7b-beta.Q4_K_M.gguf",
                    label="name of Hugging Face model file",
                    show_label=True,
                    elem_id="quantised model file step 3 input",
                )
                quantised_model_type_step3_input = gradio.Textbox(
                    value="mistral",
                    label="type of Hugging Face model",
                    show_label=True,
                    elem_id="quantised model type step 3 input",
                )

    with gradio.Row(elem_id="step 3"):
        run_step3_button = gradio.Button(
            value="Get Response", variant="primary", elem_id="run step 3 button"
        )
        cancel_step3_button = gradio.Button(
            value="Cancel Generation", variant="stop", elem_id="cancel step 3 button"
        )

    with gradio.Row(elem_id="step 3 outputs"):
        with gradio.Group(elem_id="step 3 left column outputs"):
            llm_response_step3_output = gradio.Textbox(
                label="answer from language model",
                show_label=True,
                interactive=False,
                elem_id="llm response step 3 output",
            )
            retrieved_context_step3_output = gradio.JSON(
                label="relevant documents",
                show_label=True,
                elem_id="retrieved context step 3 output",
            )
            llm_duration_step3_output = gradio.Number(
                label="duration of language model in seconds",
                show_label=True,
                interactive=False,
                elem_id="llm duration step 3 output",
                precision=2,
            )

        with gradio.Group(elem_id="step 3 right column outputs"):
            llm_prompt_step3_output = gradio.Markdown(
                label="prompt used by language model",
                show_label=True,
                elem_id="llm prompt step 3 output",
            )

    response_generation_event = run_step3_button.click(
        answer_query,
        inputs=[
            query_step3_input,
            embedding_model_step3_input,
            database_path_step3_input,
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
    _ = cancel_step3_button.click(lambda: None, cancels=[response_generation_event])

    return database_path_step3_input, embedding_model_step3_input


def main() -> None:
    """Orchestrate entire application to query package documentation."""
    with gradio.Blocks(analytics_enabled=False, title=TITLE) as gui_application:
        with gradio.Tabs(selected=1, elem_id="application"):
            with gradio.Tab(label="Summary", id=0, elem_id="summary tab") as summary_tab:
                summary_tab_flow()

                procced_to_step1_button = gradio.Button(
                    value="Proceed to Dataset Generation",
                    variant="secondary",
                    elem_id="proceed to step 1 button",
                )

            with gradio.Tab(
                label="Dataset Generation", interactive=False, id=1, elem_id="step 1 tab"
            ) as step1_tab:
                dataset_path_step1_output = step1_tab_flow()

                with gradio.Row(elem_id="step 1 tab buttons"):
                    back_to_summary_button = gradio.Button(
                        value="Back to Summary",
                        variant="secondary",
                        interactive=True,
                        elem_id="back to summary tab button",
                    )
                    procced_to_step2_button = gradio.Button(
                        value="Proceed to Database Generation",
                        variant="secondary",
                        interactive=False,
                        elem_id="proceed to step 2 tab button",
                    )

            with gradio.Tab(
                label="Database Generation", interactive=False, id=2, elem_id="step 2 tab"
            ) as step2_tab:
                (
                    dataset_path_step2_input,
                    embedding_model_step2_output,
                    database_path_step2_output,
                ) = step2_tab_flow()

                with gradio.Row(elem_id="step 2 tab buttons"):
                    back_to_step1_button = gradio.Button(
                        value="Back to Dataset Generation",
                        variant="secondary",
                        interactive=True,
                        elem_id="back to step 1 tab button",
                    )
                    procced_to_step3_button = gradio.Button(
                        value="Proceed to Response Generation",
                        variant="secondary",
                        interactive=False,
                        elem_id="proceed to step 3 tab button",
                    )

            with gradio.Tab(
                label="Response Generation", interactive=False, id=3, elem_id="step 3 tab"
            ) as step3_tab:
                database_path_step3_input, embedding_model_step3_input = step3_tab_flow()

                back_to_step2_button = gradio.Button(
                    value="Back to Database Generation",
                    variant="secondary",
                    interactive=True,
                    elem_id="back to step 2 tab button",
                )

        procced_to_step1_button.click(switch_tab, outputs=[summary_tab, step1_tab])

        dataset_path_step1_output.change(activate_button, outputs=[procced_to_step2_button])

        procced_to_step2_button.click(switch_tab, outputs=[step1_tab, step2_tab]).success(
            update_textbox_value,
            inputs=[dataset_path_step1_output],
            outputs=[dataset_path_step2_input],
        )
        back_to_summary_button.click(switch_tab, outputs=[step1_tab, summary_tab])

        database_path_step2_output.change(activate_button, outputs=[procced_to_step3_button])

        procced_to_step3_button.click(switch_tab, outputs=[step2_tab, step3_tab]).success(
            update_textbox_value,
            inputs=[database_path_step2_output],
            outputs=[database_path_step3_input],
        ).success(
            update_textbox_value,
            inputs=[embedding_model_step2_output],
            outputs=[embedding_model_step3_input],
        )
        back_to_step1_button.click(switch_tab, outputs=[step2_tab, step1_tab])

        back_to_step2_button.click(switch_tab, outputs=[step3_tab, step2_tab])

    gui_application.queue(api_open=False, max_size=2)
    gui_application.launch(
        share=False, debug=False, max_threads=3, show_error=True, show_api=False
    )


if __name__ == "__main__":
    main()
