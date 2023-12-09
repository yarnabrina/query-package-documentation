import itertools
import json
import pathlib

import pydantic

from .step_1_generation import (
    get_all_member_details,
    get_all_module_members,
    get_all_package_contents,
)
from .step_2_generation import (
    generate_member_dataset,
    generate_module_dataset,
    generate_package_dataset,
)
from .utils_generation import Document, JSONDataset, JSONDocument, MemberDetails, Module


@pydantic.validate_call(validate_return=True)
def generate_raw_dataset(package_name: str) -> list[Document]:
    all_package_contents = get_all_package_contents(package_name)

    all_module_members: list[Module] = []
    for package_contents in all_package_contents:
        for module in package_contents.children_modules_names:
            try:
                module_members = get_all_module_members(
                    f"{package_contents.package_qualified_name}.{module}"
                )
            except ImportError:
                continue

            all_module_members.append(module_members)

    all_member_details: list[MemberDetails] = []
    for module_members in all_module_members:
        for member in module_members.module_members:
            try:
                member_details = get_all_member_details(
                    module_members.module_qualified_name, member.member_name, member.member_object
                )
            except (TypeError, ValueError):
                continue

            all_member_details.append(member_details)

    package_dataset = map(generate_package_dataset, all_package_contents)
    module_dataset = map(generate_module_dataset, all_module_members)
    member_dataset = map(generate_member_dataset, all_member_details)

    combined_dataset = itertools.chain(*package_dataset, *module_dataset, *member_dataset)

    return list(combined_dataset)


@pydantic.validate_call(validate_return=True)
def generate_json_dataset(raw_dataset: list[Document]) -> JSONDataset:
    json_dataset = [
        JSONDocument.model_validate(
            {
                "question": document.question,
                "answer": document.answer,
                "retrieval_context": document.retrieval_context,
                "tuning_prompt": document.tuning_prompt,
            }
        )
        for document in raw_dataset
    ]

    return JSONDataset.model_validate({"documents": json_dataset})


@pydantic.validate_call
def store_json_dataset(json_dataset: JSONDataset, file_path: pathlib.Path) -> None:
    with pathlib.Path(file_path).open(mode="w", encoding="utf-8") as file_object:
        json.dump(json_dataset.model_dump(), file_object, indent=4)


@pydantic.validate_call(validate_return=True)
def load_json_dataset(file_path: pathlib.Path) -> JSONDataset:
    with pathlib.Path(file_path).open(mode="r", encoding="utf-8") as file_object:
        json_dataset = json.load(file_object)

    return JSONDataset.model_validate(json_dataset)


__all__ = [
    "generate_json_dataset",
    "generate_raw_dataset",
    "load_json_dataset",
    "store_json_dataset",
]
