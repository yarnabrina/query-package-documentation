"""Define functionalities to orchestrate dataset generation."""

import itertools
import json
import logging
import pathlib

import pydantic

from .step_1_generation import (
    get_all_member_details,
    get_all_module_contents,
    get_all_package_contents,
)
from .step_2_generation import (
    generate_member_dataset,
    generate_module_dataset,
    generate_package_dataset,
)
from .utils_generation import Dataset, JSONDataset, JSONDocument, MemberDetails, ModuleDetails

LOGGER = logging.getLogger(__name__)


@pydantic.validate_call(validate_return=True)
def generate_raw_datasets(package_name: str) -> list[Dataset]:
    """Generate all retrieval and tuning documents for exploring documentation of a package.

    Parameters
    ----------
    package_name : str
        name of the root package to import with

    Returns
    -------
    list[Dataset]
        all retrieval and tuning documents for root package and its contents
    """
    all_package_contents = get_all_package_contents(package_name)
    LOGGER.info(f"Enlisted total {len(all_package_contents)} packages recursively.")

    all_module_contents: list[ModuleDetails] = []
    for package_contents in all_package_contents:
        for module in package_contents.children_modules_names:
            try:
                module_contents = get_all_module_contents(
                    f"{package_contents.package_qualified_name}.{module}"
                )
            except ImportError:
                LOGGER.warning(f"Failed to import {module=}.")

                continue

            all_module_contents.append(module_contents)

    LOGGER.info(f"Enlisted total {len(all_module_contents)} modules recursively.")

    all_member_details: list[MemberDetails] = []
    for module_contents in all_module_contents:
        for member in module_contents.module_members:
            try:
                member_details = get_all_member_details(
                    module_contents.module_qualified_name, member.member_name, member.member_object
                )
            except (TypeError, ValueError):
                continue

            all_member_details.append(member_details)

    LOGGER.info(f"Enlisted total {len(all_member_details)} members recursively.")

    package_datasets = map(generate_package_dataset, all_package_contents)
    module_datasets = map(generate_module_dataset, all_module_contents)
    member_datasets = map(generate_member_dataset, all_member_details)

    combined_datasets = itertools.chain(package_datasets, module_datasets, *member_datasets)

    return list(combined_datasets)


@pydantic.validate_call(validate_return=True)
def generate_json_dataset(raw_datasets: list[Dataset]) -> JSONDataset:
    """Convert raw documents into JSON format.

    Parameters
    ----------
    raw_datasets : list[Dataset]
        all retrieval and tuning documents for root package and its contents

    Returns
    -------
    JSONDataset
        all details for querying a package documentation in JSON format
    """
    retrieval_documents: list[str] = []
    tuning_documents: list[JSONDocument] = []

    for dataset in raw_datasets:
        retrieval_documents.extend(dataset.retrieval_chunks)

        tuning_documents.extend(
            [
                JSONDocument.model_validate(document.model_dump())
                for document in dataset.tuning_documents
            ]
        )

    return JSONDataset.model_validate(
        {"retrieval_documents": retrieval_documents, "tuning_documents": tuning_documents}
    )


@pydantic.validate_call
def store_json_dataset(json_dataset: JSONDataset, file_path: pathlib.Path) -> None:
    """Dump JSON dataset into a JSON file.

    Parameters
    ----------
    json_dataset : JSONDataset
        all details for querying a package documentation in JSON format
    file_path : pathlib.Path
        path to store JSON dataset
    """
    with pathlib.Path(file_path).open(mode="w", encoding="utf-8") as file_object:
        json.dump(json_dataset.model_dump(), file_object, indent=4)


@pydantic.validate_call(validate_return=True)
def load_json_dataset(file_path: pathlib.Path) -> JSONDataset:
    """Load JSON dataset from a JSON file.

    Parameters
    ----------
    file_path : pathlib.Path
        path to load JSON dataset from

    Returns
    -------
    JSONDataset
        all details for querying a package documentation in JSON format
    """
    with pathlib.Path(file_path).open(mode="r", encoding="utf-8") as file_object:
        json_dataset = json.load(file_object)

    return JSONDataset.model_validate(json_dataset)


__all__ = [
    "generate_json_dataset",
    "generate_raw_datasets",
    "load_json_dataset",
    "store_json_dataset",
]
