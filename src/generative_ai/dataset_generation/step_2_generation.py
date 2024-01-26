"""Define functionalities to generate retrieval and tuning sources."""

import inspect
import logging
import random

import pydantic

from .utils_generation import (
    ClassDetails,
    Dataset,
    EnumDetails,
    FunctionDetails,
    MemberDetails,
    MemberType,
    ModuleDetails,
    PackageDetails,
    SplitName,
    SplitProportions,
)

random.seed(a=0)

LOGGER = logging.getLogger(__name__)

DEFAULT_SPLIT_PROPORTIONS = SplitProportions(
    train_proportion=0.6, validation_proportion=0.2, test_proportion=0.2
)


@pydantic.validate_call(validate_return=True)
def allocate_tuning_pairs(
    tuning_pairs: list[tuple[str, str]],
    split_proportions: SplitProportions = DEFAULT_SPLIT_PROPORTIONS,
) -> list[tuple[str, str, SplitName]]:
    """Allocate tuning pairs to different splits.

    Parameters
    ----------
    tuning_pairs : list[tuple[str, str]]
        question and answer pairs to be allocated to different splits
    split_proportions : SplitProportions, optional
        chance of a pair to be allocated to different splits, by default DEFAULT_SPLIT_PROPORTIONS

    Returns
    -------
    list[tuple[str, str, SplitName]]
        updated tuning pairs with split allocation
    """
    allocations = random.choices(  # noqa: S311
        [SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST],
        weights=[
            split_proportions.train_proportion,
            split_proportions.validation_proportion,
            split_proportions.test_proportion,
        ],
        k=len(tuning_pairs),
    )

    return [
        (question, answer, allocation)
        for (question, answer), allocation in zip(tuning_pairs, allocations, strict=True)
    ]


@pydantic.validate_call(validate_return=True)
def enumerate_array_elements(array: list, attribute: str | None = None) -> str:
    """Store all members of ``array``, or a common property of them.

    Parameters
    ----------
    array : list
        original objects whose elements (or their property) are to be stored
    attribute : str | None, optional
        name of common property of ``array`` elements that need to be stored, by default None

    Returns
    -------
    str
        concatenated string with all elements of ``array`` (or their property) in a numbered list

    Raises
    ------
    ValueError
        if elements of ``array`` are not strings and ``attribute`` is missing
    """
    elements = []
    for element in array:
        if isinstance(element, str):
            elements.append(element)
        elif attribute is not None:
            elements.append(getattr(element, attribute))
        else:
            LOGGER.error(f"Received {attribute=} along with {array=}")

            raise ValueError("attribute must be non-null if array elements are not string")

    return " ".join(f"{counter + 1}. {element}" for counter, element in enumerate(elements))


@pydantic.validate_call(validate_return=True)
def generate_package_dataset(package_contents: PackageDetails) -> Dataset:  # noqa: PLR0915
    """Create relevant question and answers based on package details.

    Parameters
    ----------
    package_contents : PackageDetails
        details of a python package

    Returns
    -------
    Dataset
        all documents for retrieval and tuning for querying package documentation
    """
    package_name = package_contents.package_name
    package_full_name = package_contents.package_qualified_name

    package = f"'{package_name}' package"

    package_retrieval_chunks: list[str] = [f"'{package_name}' is a Python package."]
    package_tuning_pairs: list[tuple[str, str, SplitName]] = []

    if (parent_package := package_contents.parent_package_name) is None:
        root_package_pairs = [
            ("What is the root package?", f"'{package_name}' is the root package."),
            (
                "Can you tell me what the root package is?",
                f"Sure, the root package is '{package_name}'.",
            ),
            (
                "I'm trying to find out the root package. Can you help?",
                f"Of course, the root package is '{package_name}'.",
            ),
            (
                "Do you know what the root package is?",
                f"Yes, the root package is '{package_name}'.",
            ),
            (
                "I'd like to know the root package.",
                f"The root package you're asking about is '{package_name}'.",
            ),
            (
                "Could you identify the root package?",
                f"Certainly, '{package_name}' is the root package.",
            ),
        ]
        package_retrieval_chunks.append(f"'{package_name}' is the root package.")
        package_tuning_pairs.extend(allocate_tuning_pairs(root_package_pairs))

        parent_package_pairs = [
            (
                f"Name parent package of '{package_name}'.",
                f"Being the root package, '{package_name}' has no parent package.",
            ),
            (
                f"What is the parent package of '{package_name}'?",
                f"The root package '{package_name}' does not have a parent package.",
            ),
            (
                f"Can you tell me the parent package of '{package_name}'?",
                f"'{package_name}' is a root package and therefore,"
                " it does not have a parent package.",
            ),
            (
                f"Could you identify the parent package of '{package_name}'?",
                f"As a root package, '{package_name}' does not possess a parent package.",
            ),
            (
                f"I'm looking for the parent package of '{package_name}'. Can you help?",
                f"Sure, '{package_name}' is a root package, so it doesn't have a parent package.",
            ),
            (
                f"Do you know the parent package of '{package_name}'?",
                f"Yes, '{package_name}' is a root package and hence,"
                " it doesn't have a parent package.",
            ),
        ]
        package_retrieval_chunks.append(f"'{package_name}' has no parent package.")
        package_tuning_pairs.extend(allocate_tuning_pairs(parent_package_pairs))
    else:
        parent_package_pairs = [
            (
                f"Name parent package of '{package_name}' sub-package.",
                f"'{parent_package}' is the full name of its parent package.",
            ),
            (
                f"What is the parent package of the '{package_name}' sub-package?",
                f"The parent package of '{package_name}' is '{parent_package}'.",
            ),
            (
                f"Could you tell me the parent package of '{package_name}'?",
                f"Sure, the parent package of '{package_name}' is '{parent_package}'.",
            ),
            (
                f"I need to know the parent package of '{package_name}'.",
                f"The parent package of '{package_name}' is '{parent_package}'.",
            ),
            (
                f"Identify the parent package for the '{package_name}' sub-package.",
                f"The parent package for '{package_name}' is identified as '{parent_package}'.",
            ),
            (
                f"Can you name the parent package of the '{package_name}' sub-package?",
                f"Yes, the parent package of '{package_name}' is '{parent_package}'.",
            ),
        ]
        package_retrieval_chunks.append(
            f"'{package_name}' is part of parent package '{parent_package}'."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(parent_package_pairs))

        package_full_name_pairs = [
            (
                f"Tell the full name of '{package_name}' sub-package.",
                f"'{package_full_name}' is the fully qualified name of '{package_name}'.",
            ),
            (
                f"What is the fully qualified name of the '{package_name}' sub-package?",
                f"Fully qualified name of '{package_name}' sub-package is '{package_full_name}'.",
            ),
            (
                f"Could you provide the full name of the '{package_name}' sub-package?",
                f"Sure, the full name of '{package_name}' sub-package is '{package_full_name}'.",
            ),
            (
                f"I need the full name of the '{package_name}' sub-package. Can you tell me?",
                f"Of course, full name of '{package_name}' sub-package is '{package_full_name}'.",
            ),
            (
                f"Can you inform me about the full name of the '{package_name}' sub-package?",
                f"Certainly, full name of '{package_name}' sub-package is '{package_full_name}'.",
            ),
            (
                f"Please, reveal the full name of the '{package_name}' sub-package.",
                f"Absolutely, full name of '{package_name}' sub-package is '{package_full_name}'.",
            ),
        ]
        package_retrieval_chunks.append(
            f"Full name of '{package_name}' sub-package is '{package_full_name}'."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_full_name_pairs))

        package_hierarchy = enumerate_array_elements(package_contents.package_hierarchy)
        package_hierarchy_pairs = [
            (
                f"What is the hierarchy of {package}?",
                f"The hierarchy of {package} is as follows: {package_hierarchy}.",
            ),
            (
                f"Can you explain the hierarchy of the {package}?",
                f"Sure, the hierarchy of the {package} is: {package_hierarchy}.",
            ),
            (
                f"Could you describe the structure of the {package}?",
                f"Of course, the structure of {package} is: {package_hierarchy}.",
            ),
            (
                f"I need to understand the hierarchy of {package}. Can you help?",
                f"Absolutely, the hierarchy of {package} is: {package_hierarchy}.",
            ),
            (
                f"Please provide the hierarchy of the {package}.",
                f"The hierarchy of the {package} is: {package_hierarchy}.",
            ),
            (
                f"I'm interested in the structure of the {package}. What is it?",
                f"The structure of {package} is as follows: {package_hierarchy}.",
            ),
        ]
        package_retrieval_chunks.append(
            f"Hierarchy of {package} is as follows: {package_hierarchy}."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_hierarchy_pairs))

    if not (children_sub_packages := package_contents.children_sub_packages_names):
        package_sub_package_pairs = [
            (
                f"List the sub-packages of {package}.",
                f"{package} does not have any further sub-packages.",
            ),
            (
                f"What are the sub-packages of the {package}?",
                f"The {package} does not contain any sub-packages.",
            ),
            (
                f"Could you tell me the sub-packages of {package}?",
                f"I'm sorry, but the {package} doesn't have any sub-packages.",
            ),
            (
                f"I need to know the sub-packages of {package}. Can you list them?",
                f"Unfortunately, {package} doesn't include any sub-packages.",
            ),
            (
                f"Can you provide a list of sub-packages for the {package}?",
                f"There are no sub-packages in the {package}.",
            ),
            (
                f"Identify the sub-packages of {package}.",
                f"No sub-packages are present in the {package}.",
            ),
        ]
        package_retrieval_chunks.append(f"{package} does not have any further sub-packages.")
        package_tuning_pairs.extend(allocate_tuning_pairs(package_sub_package_pairs))
    else:
        children_sub_packages_count = len(children_sub_packages)
        children_sub_packages_count_pairs = [
            (
                f"How many sub-packages are there in {package}?",
                f"{package} has {children_sub_packages_count} many sub-packages.",
            ),
            (
                f"What is the count of sub-packages in {package}?",
                f"The count of sub-packages in {package} is {children_sub_packages_count}.",
            ),
            (
                f"Could you tell me the number of sub-packages available in {package}?",
                f"{package} has {children_sub_packages_count} sub-packages.",
            ),
            (
                f"Please provide the count of sub-packages for {package}.",
                f"Number of sub-packages in {package} is {children_sub_packages_count}.",
            ),
            (
                f"Tell me the quantity of sub-packages present in {package}.",
                f"{package} has {children_sub_packages_count} sub-packages.",
            ),
            (
                f"Would you mind letting me know how many sub-packages {package} contains?",
                f"{package} contains {children_sub_packages_count} sub-packages.",
            ),
        ]
        package_retrieval_chunks.append(
            f"{package} has {children_sub_packages_count} many sub-packages."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(children_sub_packages_count_pairs))

        package_sub_packages = enumerate_array_elements(children_sub_packages)
        package_sub_package_pairs = [
            (
                f"List the sub-packages of {package}.",
                f"Sub-packages of {package} are as follows: {package_sub_packages}.",
            ),
            (
                f"What are the sub-packages of the {package}?",
                f"The {package} has the following sub-packages: {package_sub_packages}.",
            ),
            (
                f"Could you tell me the sub-packages of {package}?",
                f"Sure, the sub-packages of {package} are: {package_sub_packages}.",
            ),
            (
                f"I need to know the sub-packages of {package}. Can you list them?",
                f"Of course, the sub-packages of {package} are: {package_sub_packages}.",
            ),
            (
                f"Please provide the sub-packages of {package}.",
                f"The sub-packages of {package} are: {package_sub_packages}.",
            ),
            (
                f"Can you enumerate the sub-packages of {package}?",
                f"Certainly, the sub-packages of {package} are: {package_sub_packages}.",
            ),
        ]
        package_retrieval_chunks.append(
            f"Sub-packages of {package} are as follows: {package_sub_packages}."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_sub_package_pairs))

    if not (children_modules := package_contents.children_modules_names):
        package_module_pairs = [
            (
                f"What are the modules of {package}?",
                f"{package} does not have any direct modules under itself.",
            ),
            (
                f"Can you list the modules under the {package}?",
                f"There are no direct modules under the {package}.",
            ),
            (
                f"Does the {package} contain any modules?",
                f"No, the {package} does not contain any direct modules.",
            ),
            (
                f"I'm looking for the modules of {package}. Can you help?",
                f"I'm sorry, but {package} does not have any direct modules.",
            ),
            (
                f"Tell me about the modules of {package}.",
                f"Actually, the {package} does not have any direct modules.",
            ),
            (
                f"Are there any modules under the {package}?",
                f"No, there aren't any direct modules under the {package}.",
            ),
        ]
        package_retrieval_chunks.append(f"{package} does not have any further modules.")
        package_tuning_pairs.extend(allocate_tuning_pairs(package_module_pairs))
    else:
        children_modules_count = len(children_modules)
        children_modules_count_pairs = [
            (
                f"How many modules are there in {package}?",
                f"{package} has {children_modules_count} many modules.",
            ),
            (
                f"What is the count of modules in {package}?",
                f"The count of modules in {package} is {children_modules_count}.",
            ),
            (
                f"Could you tell me the number of modules available in {package}?",
                f"{package} has {children_modules_count} modules.",
            ),
            (
                f"Please provide the count of modules for {package}.",
                f"The number of modules in {package} is {children_modules_count}.",
            ),
            (
                f"Tell me the quantity of modules present in {package}.",
                f"{package} has {children_modules_count} modules.",
            ),
            (
                f"Would you mind letting me know how many modules {package} contains?",
                f"{package} contains {children_modules_count} modules.",
            ),
        ]
        package_retrieval_chunks.append(f"{package} has {children_modules_count} many modules.")
        package_tuning_pairs.extend(allocate_tuning_pairs(children_modules_count_pairs))

        package_modules = enumerate_array_elements(children_modules)
        package_module_pairs = [
            (
                f"What are the modules of {package}?",
                f"Direct modules under {package} are as follows: {package_modules}.",
            ),
            (
                f"Can you list the modules of the {package}?",
                f"Sure, the direct modules under {package} are: {package_modules}.",
            ),
            (
                f"I need to know the modules of the {package}.",
                f"The modules you're looking for in {package} are: {package_modules}.",
            ),
            (
                f"Could you tell me what the modules of the {package} are?",
                f"Of course, the modules under {package} are: {package_modules}.",
            ),
            (
                f"I'm interested in the modules of the {package}.",
                f"The modules in {package} are: {package_modules}.",
            ),
            (
                f"What modules does the {package} contain?",
                f"The {package} contains these modules: {package_modules}.",
            ),
        ]
        package_retrieval_chunks.append(f"Modules of {package} are as follows: {package_modules}.")
        package_tuning_pairs.extend(allocate_tuning_pairs(package_module_pairs))

    if not (package_summary := package_contents.package_summary):
        package_summary_pairs = [
            (f"What does {package} do?", f"{package} does not have any documentation."),
            (
                f"Can you tell me the functionality of the {package}?",
                f"Unfortunately, the {package} provides no documentation.",
            ),
            (
                f"I'm curious about what the {package} does. Can you enlighten me?",
                f"I'm sorry, but the {package} does not come with any documentation.",
            ),
            (
                f"Could you explain the purpose of the {package}?",
                f"Regrettably, the {package} lacks any form of documentation.",
            ),
            (
                f"What's the role of the {package}?",
                f"The {package} does not offer any documentation.",
            ),
            (
                f"What functionality does the {package} provide?",
                f"The {package} does not have any available documentation.",
            ),
        ]
        package_retrieval_chunks.append(
            f"Unfortunately, {package} currently does not have any documentation."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_summary_pairs))
    else:
        package_summary_pairs = [
            (f"What does {package} do?", f"Its documentation is as follows: '{package_summary}'."),
            (
                f"Can you tell me about the {package}?",
                f"Sure, here is its documentation: '{package_summary}'.",
            ),
            (
                f"I'd like to know what the {package} does.",
                f"Of course, here's the documentation for it: '{package_summary}'.",
            ),
            (
                f"Could you explain the functionality of the {package}?",
                f"Absolutely, the documentation states: '{package_summary}'.",
            ),
            (
                f"What's the purpose of the {package}?",
                f"The purpose is described in its documentation: '{package_summary}'.",
            ),
            (
                f"I'm curious about the {package}, what does it do?",
                f"Good question, its documentation reads: '{package_summary}'.",
            ),
        ]
        package_retrieval_chunks.append(
            f"The following is the documentation of {package}: '{package_summary}'."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_summary_pairs))

    if not (package_exports := package_contents.package_all_exports):
        package_members_pairs = [
            (
                f"What are the public members of the {package}?",
                f"{package} does not have any public member exported through '__all__'.",
            ),
            (
                f"Can you list the public members of the {package}?",
                f"The {package} does not export any public members through '__all__'.",
            ),
            (
                f"Are there any public members in the {package}?",
                f"No, the {package} does not have any public members exported through '__all__'.",
            ),
            (
                f"I'm looking for public members of {package}. Can you help?",
                f"Sure, but the {package} does not export any public members through '__all__'.",
            ),
            (
                f"Could you tell me the public members of the {package}?",
                f"Unfortunately, the {package} does not have any public members"
                " exported through '__all__'.",
            ),
            (
                f"I'd like to know the public members of the {package}."
                " Can you provide that information?",
                f"I'm sorry, but the {package} does not have any public members"
                " exported through '__all__'.",
            ),
        ]
        package_retrieval_chunks.append(
            f"{package} does not export anything publicly using __all__ variable."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_members_pairs))
    else:
        package_exports_count = len(package_exports)
        package_exports_count_pairs = [
            (
                f"How many objects does {package} export publicly?",
                f"{package} exports {package_exports_count} many objects using __all__.",
            ),
            (
                f"What is the count of publicly exported objects in {package}?",
                f"Count of publicly exported objects in {package} is {package_exports_count}.",
            ),
            (
                f"Could you tell me the number of objects publicly exported by {package}?",
                f"{package} exports {package_exports_count} objects using __all__.",
            ),
            (
                f"Please provide the count of objects publicly exported by {package}.",
                f"Number of objects publicly exported by {package} is {package_exports_count}.",
            ),
            (
                f"Tell me the quantity of objects that {package} exports publicly.",
                f"{package} exports {package_exports_count} objects using __all__.",
            ),
            (
                f"Would you mind letting me know how many objects {package} publicly exports?",
                f"{package} publicly exports {package_exports_count} objects.",
            ),
        ]
        package_retrieval_chunks.append(
            f"{package} has {package_exports_count} many public exports."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_exports_count_pairs))

        package_public_members = enumerate_array_elements(package_exports)
        package_members_pairs = [
            (
                f"What are the public members of the {package}?",
                f"{package} publicly exports the following members using '__all__':"
                f" {package_public_members}.",
            ),
            (
                f"Can you list the public members of the {package}?",
                f"Sure, the {package} publicly exports these members using '__all__':"
                f" {package_public_members}.",
            ),
            (
                f"I need to know the public members of the {package}. Can you tell me?",
                f"Of course, the {package} publicly exports these members using '__all__':"
                f" {package_public_members}.",
            ),
            (
                f"Could you tell me what the {package} publicly exports?",
                f"The {package} publicly exports the following members using '__all__':"
                f" {package_public_members}.",
            ),
            (
                f"I'm interested in the public members of the {package}. What are they?",
                f"The {package} publicly exports these members using '__all__':"
                f" {package_public_members}.",
            ),
        ]
        package_retrieval_chunks.append(
            f"{package} exports following public members using __all__: {package_public_members}."
        )
        package_tuning_pairs.extend(allocate_tuning_pairs(package_members_pairs))

    package_dataset = Dataset(
        retrieval_chunks=package_retrieval_chunks, tuning_pairs=package_tuning_pairs
    )

    return package_dataset


@pydantic.validate_call(validate_return=True)
def generate_module_dataset(module_contents: ModuleDetails) -> Dataset:
    """Create relevant question and answers based on module details.

    Parameters
    ----------
    module_contents : ModuleDetails
        details of a python module

    Returns
    -------
    Dataset
        all documents for retrieval and tuning for querying module documentation
    """
    module_name = module_contents.module_name
    module_full_name = module_contents.module_qualified_name
    module = f"'{module_name}' module"

    module_retrieval_chunks: list[str] = [f"'{module_name}' is a Python module."]
    module_tuning_pairs: list[tuple[str, str, SplitName]] = []

    module_package_pairs = [
        (
            f"Can you tell the the parent package of {module}?",
            f"'{module_contents.package_name}' is the parent package of {module}.",
        ),
        (
            f"What is the parent package of the {module}?",
            f"The parent package of {module} is '{module_contents.package_name}'.",
        ),
        (
            f"I'm trying to find the parent package of the {module}. Can you help?",
            f"Sure, parent package of {module} is '{module_contents.package_name}'.",
        ),
        (
            f"Could you inform me about the parent package of the {module}?",
            f"Certainly, '{module_contents.package_name}' is the parent package of the {module}.",
        ),
        (
            f"I need to know the parent package of {module}. Can you provide that information?",
            f"Absolutely, the parent package of the {module} is '{module_contents.package_name}'.",
        ),
        (
            f"Can you identify the parent package for the {module}?",
            f"Yes, parent package for {module} is '{module_contents.package_name}'.",
        ),
    ]
    module_retrieval_chunks.append(
        f"{module} is part of parent package '{module_contents.package_name}'."
    )
    module_tuning_pairs.extend(allocate_tuning_pairs(module_package_pairs))

    module_full_name_pairs = [
        (
            f"Specify the full name of {module}?",
            f"'{module_full_name}' is fully qualified name for {module}.",
        ),
        (
            f"What is the fully qualified name for the {module}?",
            f"The fully qualified name for the {module} is '{module_full_name}'.",
        ),
        (
            f"Could you tell me the full name of the {module}?",
            f"Sure, the full name of the {module} is '{module_full_name}'.",
        ),
        (
            f"I need the full name of the {module}. Can you provide it?",
            f"Of course, the full name of the {module} is '{module_full_name}'.",
        ),
        (
            f"Can you specify the fully qualified name of the {module}?",
            f"Yes, fully qualified name of the {module} is '{module_full_name}'.",
        ),
        (
            f"I'm looking for the full name of the {module}. What is it?",
            f"Full name of the {module} you're looking for is '{module_full_name}'.",
        ),
    ]
    module_retrieval_chunks.append(f"Full name of {module} is '{module_full_name}'.")
    module_tuning_pairs.extend(allocate_tuning_pairs(module_full_name_pairs))

    module_hierarchy = enumerate_array_elements(module_contents.module_hierarchy)
    module_hierarchy_pairs = [
        (
            f"What is the hierarchy of {module}?",
            f"The hierarchy of {module} is as follows: {module_hierarchy}.",
        ),
        (
            f"Can you explain the hierarchy of the {module}?",
            f"Sure, the hierarchy of the {module} is: {module_hierarchy}.",
        ),
        (
            f"Could you describe the structure of the {module}?",
            f"Of course, the structure of the {module} is: {module_hierarchy}.",
        ),
        (
            f"I need to understand the hierarchy of the {module}. Can you help?",
            f"Absolutely, the hierarchy of the {module} is: {module_hierarchy}.",
        ),
        (
            f"Please provide the hierarchy of the {module}.",
            f"The hierarchy of the {module} is: {module_hierarchy}.",
        ),
        (
            f"What does the hierarchy of the {module} look like?",
            f"The hierarchy of the {module} looks like this: {module_hierarchy}.",
        ),
    ]
    module_retrieval_chunks.append(f"Hierarchy of {module} is as follows: {module_hierarchy}.")
    module_tuning_pairs.extend(allocate_tuning_pairs(module_hierarchy_pairs))

    module_members_count = len(module_contents.module_members)
    module_members_count_pairs = [
        (
            f"How many members does {module} have?",
            f"{module} has {module_members_count} many members.",
        ),
        (
            f"What is the count of members in {module}?",
            f"The count of members in {module} is {module_members_count}.",
        ),
        (
            f"Could you tell me the number of members in {module}?",
            f"{module} has {module_members_count} members.",
        ),
        (
            f"Please provide the count of members for {module}.",
            f"The number of members in {module} is {module_members_count}.",
        ),
        (
            f"Tell me the quantity of members present in {module}.",
            f"{module} has {module_members_count} members.",
        ),
        (
            f"Would you mind letting me know how many members {module} contains?",
            f"{module} contains {module_members_count} members.",
        ),
    ]
    module_retrieval_chunks.append(f"{module} has {module_members_count} many members.")
    module_tuning_pairs.extend(allocate_tuning_pairs(module_members_count_pairs))

    module_member_names = enumerate_array_elements(
        module_contents.module_members, attribute="member_name"
    )
    module_members_pairs = [
        (
            f"List the members of {module}.",
            f"Members of {module} are as follows: {module_member_names}.",
        ),
        (
            f"What are the members of the {module}?",
            f"The {module} has the following members: {module_member_names}.",
        ),
        (
            f"Can you tell me the members of the {module}?",
            f"Sure, the members of the {module} are: {module_member_names}.",
        ),
        (
            f"I need to know the members of the {module}.",
            f"Members of {module} you asked for are: {module_member_names}.",
        ),
        (
            f"Could you list the members of the {module}?",
            f"Of course, members of the {module} are: {module_member_names}.",
        ),
        (
            f"Please provide the members of the {module}.",
            f"Members of {module} you requested are: {module_member_names}.",
        ),
    ]
    module_retrieval_chunks.append(f"Members of {module} are as follows: {module_member_names}.")
    module_tuning_pairs.extend(allocate_tuning_pairs(module_members_pairs))

    if not (module_summary := module_contents.module_summary):
        module_summary_pairs = [
            (f"What is the {module} for?", f"{module} does not have any documentation."),
            (
                f"Can you tell me the purpose of the {module}?",
                f"The {module} lacks any documentation.",
            ),
            (
                f"I'd like to know what the {module} is used for.",
                f"Unfortunately, there is no documentation for the {module}.",
            ),
            (
                f"Could you explain the function of the {module}?",
                f"Regrettably, the {module} doesn't come with any documentation.",
            ),
            (f"What does the {module} do?", f"The {module} is without any documentation."),
        ]
        module_retrieval_chunks.append(
            f"Unfortunately, {module} currently does not have any documentation."
        )
        module_tuning_pairs.extend(allocate_tuning_pairs(module_summary_pairs))
    else:
        module_summary_pairs = [
            (
                f"What is the '{module_name}' module for?",
                f"{module} documents itself as follows: '{module_summary}'.",
            ),
            (
                f"Can you tell me the purpose of the '{module_name}' module?",
                f"Purpose of {module} is documented as: '{module_summary}'.",
            ),
            (
                f"I'm curious about the '{module_name}' module. What does it do?",
                f"The {module} is described as: '{module_summary}'.",
            ),
            (
                f"Could you explain the functionality of the '{module_name}' module?",
                f"The functionality of the {module} is described as: '{module_summary}'.",
            ),
            (
                f"I'd like to know more about the '{module_name}' module. What's its role?",
                f"The role of the {module} is: '{module_summary}'.",
            ),
            (
                f"What's the use of the '{module_name}' module?",
                f"Use of the {module} is documented as: '{module_summary}'.",
            ),
        ]
        module_retrieval_chunks.append(
            f"The following is the documentation of {module}: {module_summary}."
        )
        module_tuning_pairs.extend(allocate_tuning_pairs(module_summary_pairs))

    if not (module_exports := module_contents.module_all_exports):
        module_exports_pairs = [
            (
                f"Tell me the public members of the {module}.",
                f"{module} lacks any public member exported through '__all__'.",
            ),
            (
                f"What are the public members of the {module}?",
                "There are no public members exported through '__all__' in the {module}.",
            ),
            (
                f"Could you list the public members of the {module}?",
                f"Unfortunately, {module} does not export any public members through '__all__'.",
            ),
            (
                f"I need to know the public members of the {module}.",
                f"The {module} does not have any public members exported through '__all__'.",
            ),
            (
                f"Can you show me the public members of the {module}?",
                f"The {module} does not contain any public members exported through '__all__'.",
            ),
            (
                f"I'm interested in the public members of the {module}. What are they?",
                f"{module} does not export any public members through '__all__'.",
            ),
        ]
        module_retrieval_chunks.append(
            f"{module} does not export anything publicly using __all__ variable."
        )
        module_tuning_pairs.extend(allocate_tuning_pairs(module_exports_pairs))
    else:
        module_exports_count = len(module_exports)
        module_exports_count_pairs = [
            (
                f"How many objects does {module} export publicly?",
                f"{module} exports {module_exports_count} many objects using __all__.",
            ),
            (
                f"What is the count of publicly exported objects in {module}?",
                f"The count of publicly exported objects in {module} is {module_exports_count}.",
            ),
            (
                f"Could you tell me the number of objects publicly exported by {module}?",
                f"{module} exports {module_exports_count} objects using __all__.",
            ),
            (
                f"Please provide the count of objects publicly exported by {module}.",
                f"The number of objects publicly exported by {module} is {module_exports_count}.",
            ),
            (
                f"Tell me the quantity of objects that {module} exports publicly.",
                f"{module} exports {module_exports_count} objects using __all__.",
            ),
            (
                f"Would you mind letting me know how many objects {module} publicly exports?",
                f"{module} publicly exports {module_exports_count} objects.",
            ),
        ]
        module_retrieval_chunks.append(f"{module} has {module_exports_count} many public exports.")
        module_tuning_pairs.extend(allocate_tuning_pairs(module_exports_count_pairs))

        module_public_exports = enumerate_array_elements(module_exports)
        module_exports_pairs = [
            (
                f"Tell me the public members of the {module}.",
                f"{module} publicly exports the following members using '__all__':"
                f" {module_public_exports}.",
            ),
            (
                f"What are the public members of the {module}?",
                f"The {module} publicly exports the following members using '__all__':"
                f" {module_public_exports}.",
            ),
            (
                f"Could you list the public members of the {module}?",
                f"Sure, the {module} publicly exports these members using '__all__':"
                f" {module_public_exports}.",
            ),
            (
                f"I need to know the public members of the {module}.",
                f"The {module} publicly exports these members using '__all__':"
                f" {module_public_exports}.",
            ),
            (
                f"Can you show me the public members of the {module}?",
                f"Of course, the {module} publicly exports the following members using '__all__':"
                f" {module_public_exports}.",
            ),
        ]
        module_retrieval_chunks.append(
            f"{module} exports following members using __all__: {module_public_exports}."
        )
        module_tuning_pairs.extend(allocate_tuning_pairs(module_exports_pairs))

    module_dataset = Dataset(
        retrieval_chunks=module_retrieval_chunks, tuning_pairs=module_tuning_pairs
    )

    return module_dataset


@pydantic.validate_call(validate_return=True)
def generate_enum_member_dataset(
    enum_member: str, enum_docstring: str, member_type_details: EnumDetails
) -> tuple[Dataset, list[str]]:
    """Create relevant question and answers based on enum member details.

    Parameters
    ----------
    enum_member : str
        name of the enum member
    enum_docstring : str
        ``__doc__`` attribute of the enum member, if any
    member_type_details : EnumDetails
        details of the enum member

    Returns
    -------
    Dataset
        all documents for retrieval and tuning for querying enum member documentation
    list[str]
        only retrieval documents
    """
    enum_member_retrieval_chunks: list[str] = [
        f"{enum_member} is a Python enum.",
        f"{enum_member} has following docstring: {enum_docstring}.",
    ]
    enum_member_tuning_pairs: list[tuple[str, str, SplitName]] = []

    enum_member_count = len(member_type_details.enum_members)
    enum_member_count_pairs = [
        (
            f"How many members are there in {enum_member}?",
            f"{enum_member} has {enum_member_count} members.",
        ),
        (
            f"What is the count of members in {enum_member}?",
            f"The count of members in {enum_member} is {enum_member_count}.",
        ),
        (
            f"Can you tell me the number of members in {enum_member}?",
            f"Sure, the number of members in {enum_member} is {enum_member_count}.",
        ),
        (
            f"Could you provide the total number of members in {enum_member}?",
            f"The total number of members in {enum_member} is {enum_member_count}.",
        ),
        (
            f"I need to know the quantity of members in {enum_member}.",
            f"The quantity of members in {enum_member} is {enum_member_count}.",
        ),
        (
            f"Please inform me about the number of members in {enum_member}.",
            f"The number of members in {enum_member} is {enum_member_count}.",
        ),
    ]
    enum_member_retrieval_chunks.insert(-1, f"{enum_member} has {enum_member_count} many members.")
    enum_member_tuning_pairs.extend(allocate_tuning_pairs(enum_member_count_pairs))

    enum_members = enumerate_array_elements(
        member_type_details.enum_members, attribute="enum_member"
    )
    enum_members_pairs = [
        (
            f"What are the different members of {enum_member}?",
            f"Different members of {enum_member} are as follows: {enum_members}.",
        ),
        (
            f"Can you list the different members of {enum_member}?",
            f"Sure, the different members of {enum_member} are: {enum_members}.",
        ),
        (
            f"Could you tell me the different members of {enum_member}?",
            f"Of course, the different members of {enum_member} include: {enum_members}.",
        ),
        (
            f"I need to know the different members of {enum_member}.",
            f"The different members of {enum_member} are: {enum_members}.",
        ),
        (
            f"What does {enum_member} consist of?",
            f"{enum_member} consists of the following members: {enum_members}.",
        ),
    ]
    enum_member_retrieval_chunks.insert(
        -1, f"Members of {enum_member} are as follows: {enum_members}."
    )
    enum_member_tuning_pairs.extend(allocate_tuning_pairs(enum_members_pairs))

    enum_member_names = enumerate_array_elements(
        member_type_details.enum_members, attribute="enum_member_name"
    )
    enum_member_names_pairs = [
        (
            f"List just the names of different members of {enum_member}.",
            f"Different members of {enum_member} have the following names: {enum_member_names}.",
        ),
        (
            f"Can you provide the names of different members of {enum_member}?",
            f"Sure, different members of {enum_member} are named as follows: {enum_member_names}.",
        ),
        (
            f"What are the names of different members of {enum_member}?",
            f"The names of different members of {enum_member} are: {enum_member_names}.",
        ),
        (
            f"I need the names of different members of {enum_member}.",
            f"The different members of {enum_member} have these names: {enum_member_names}.",
        ),
        (
            f"Could you list the names of different members of {enum_member}?",
            f"Of course, different members of {enum_member} have these names:"
            f" {enum_member_names}.",
        ),
        (
            f"Show me the names of different members of {enum_member}.",
            f"The names of different members of {enum_member} are: {enum_member_names}.",
        ),
    ]
    enum_member_retrieval_chunks.insert(
        -1, f"Names of different members of {enum_member} are as follows: {enum_member_names}."
    )
    enum_member_tuning_pairs.extend(allocate_tuning_pairs(enum_member_names_pairs))

    enum_member_values = enumerate_array_elements(
        member_type_details.enum_members, attribute="enum_member_value"
    )
    enum_member_values_pairs = [
        (
            f"Only show the different values supported by {enum_member}.",
            f"{enum_member} supports the following values: {enum_member_values}.",
        ),
        (
            f"What are the different values that {enum_member} supports?",
            f"The different values that {enum_member} supports are: {enum_member_values}.",
        ),
        (
            f"Can you list the values supported by {enum_member}?",
            f"Sure, {enum_member} supports these values: {enum_member_values}.",
        ),
        (
            f"I need to know the values supported by {enum_member}.",
            f"{enum_member} supports these values: {enum_member_values}.",
        ),
        (
            f"Could you tell me the values that {enum_member} supports?",
            f"Of course, the values that {enum_member} supports are: {enum_member_values}.",
        ),
        (
            f"Please provide the values supported by {enum_member}.",
            f"The values supported by {enum_member} are: {enum_member_values}.",
        ),
    ]
    enum_member_retrieval_chunks.insert(
        -1, f"Values of different members of {enum_member} are as follows: {enum_member_values}."
    )
    enum_member_tuning_pairs.extend(allocate_tuning_pairs(enum_member_values_pairs))

    enum_member_dataset = Dataset(
        retrieval_chunks=enum_member_retrieval_chunks, tuning_pairs=enum_member_tuning_pairs
    )

    return enum_member_dataset, enum_member_retrieval_chunks


@pydantic.validate_call(validate_return=True)
def generate_class_member_dataset(  # noqa: C901, PLR0912, PLR0915
    class_member: str, class_docstring: str, member_type_details: ClassDetails
) -> tuple[Dataset, list[str]]:
    """Create relevant question and answers based on class member details.

    Parameters
    ----------
    class_member : str
        name of the class member
    class_docstring : str
        ``__doc__`` attribute of the class member, if any
    member_type_details : ClassDetails
        details of the class member

    Returns
    -------
    Dataset
        all documents for retrieval and tuning for querying class member documentation
    list[str]
        only retrieval documents
    """
    class_member_retrieval_chunks: list[str] = [
        f"{class_member} is a Python class.",
        f"{class_member} has following docstring: {class_docstring}.",
    ]
    class_member_tuning_pairs: list[tuple[str, str, SplitName]] = []

    if not (class_parameters := member_type_details.class_parameters):
        class_parameters_pairs = [
            (
                f"What are the different parameters of {class_member}?",
                f"{class_member} needs no arguments for instantiation.",
            ),
            (
                f"Can you tell me the parameters required for {class_member}?",
                f"No parameters are required for instantiating {class_member}.",
            ),
            (
                f"What arguments do I need to instantiate {class_member}?",
                f"You don't need any arguments to instantiate {class_member}.",
            ),
            (
                f"Do I need any parameters to use {class_member}?",
                f"{class_member} can be used without any parameters.",
            ),
            (
                f"What should I pass as arguments when creating an instance of {class_member}?",
                "There's no need to pass any arguments"
                f" when creating an instance of {class_member}.",
            ),
            (
                f"Are there any parameters needed for the instantiation of {class_member}?",
                f"The instantiation of {class_member} doesn't require any parameters.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} requires no arguments for instantiation."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameters_pairs))
    else:
        class_parameter_names = enumerate_array_elements(
            class_parameters, attribute="parameter_details"
        )
        class_parameters_pairs = [
            (
                f"What are the different parameters of {class_member}?",
                f"{class_member} supports these arguments to initiate"
                f" a new instance: {class_parameter_names}.",
            ),
            (
                f"Can you list the parameters for {class_member}?",
                f"Sure, {class_member} can be initiated with these arguments:"
                f" {class_parameter_names}.",
            ),
            (
                f"I need to know the parameters of {class_member}.",
                f"The parameters to initiate a new instance of {class_member} are:"
                f" {class_parameter_names}.",
            ),
            (
                f"Tell me the parameters that {class_member} supports.",
                f"{class_member} can be initiated with these arguments: {class_parameter_names}.",
            ),
            (
                f"What arguments does {class_member} take for initialisation?",
                f"To initialise {class_member}, you can use these arguments:"
                f" {class_parameter_names}.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} requires the following arguments for initialisation:"
            f" {class_parameter_names}"
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameters_pairs))

    for class_parameter in class_parameters:
        parameter_name = class_parameter.parameter_name
        parameter = f"'{parameter_name}' argument in {class_member}"

        if (parameter_default := class_parameter.parameter_default) is inspect._empty:
            class_parameter_defaults_pairs = [
                (
                    f"Tell default value of {parameter}.",
                    f"{parameter} does not have a default value.",
                ),
                (
                    f"What is the default value of {parameter}?",
                    f"The {parameter} does not have a default value.",
                ),
                (
                    f"Could you inform me about default value of {parameter}?",
                    f"Sure, the {parameter} does not have a default value.",
                ),
                (
                    f"I need to know the default value of {parameter}. Can you help?",
                    f"Of course, the {parameter} does not have a default value.",
                ),
                (
                    f"Can you tell me if {parameter} has default value?",
                    f"No, the {parameter} does not have a default value.",
                ),
                (
                    f"I'm curious about default value of {parameter}.",
                    f"Well, the {parameter} does not have a default value.",
                ),
            ]
            class_member_retrieval_chunks.append(f"{parameter} does not have a default value.")
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameter_defaults_pairs))
        else:
            class_parameter_defaults_pairs = [
                (
                    f"Tell default value of {parameter}.",
                    f"{parameter} takes {parameter_default} by default.",
                ),
                (
                    f"What is the default value of {parameter}?",
                    f"The default value of {parameter} is {parameter_default}.",
                ),
                (
                    f"Could you inform me about default value of {parameter}?",
                    f"Sure, the default value of {parameter} is {parameter_default}.",
                ),
                (
                    f"I need to know the default value of {parameter}.",
                    f"The default value of {parameter} is {parameter_default}.",
                ),
                (
                    f"Can you provide default value of {parameter}?",
                    f"Yes, default value of {parameter} is {parameter_default}.",
                ),
                (
                    f"Please, disclose default value of {parameter}.",
                    f"Certainly, the default value of {parameter} is {parameter_default}.",
                ),
            ]
            class_member_retrieval_chunks.append(
                f"{parameter_default} is the default value of {parameter}."
            )
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameter_defaults_pairs))

        if (parameter_annotation := class_parameter.parameter_annotation) is inspect._empty:
            class_parameter_types_pairs = [
                (
                    f"Name type hint for {parameter}.",
                    f"{parameter} does not have a type annotation.",
                ),
                (
                    f"What is the type hint for {parameter}?",
                    f"There is no type annotation for the {parameter}.",
                ),
                (
                    f"Can you tell me the type hint for {parameter}?",
                    f"The {parameter} is not annotated with a type.",
                ),
                (
                    f"I'm looking for the type hint for {parameter}. Can you help?",
                    f"Sure, the {parameter} does not have a type annotation.",
                ),
                (
                    f"Could you provide the type hint for {parameter}?",
                    f"Unfortunately, {parameter} does not have type annotation.",
                ),
                (
                    f"I need to know the type hint for {parameter}.",
                    f"The {parameter} does not come with a type annotation.",
                ),
            ]
            class_member_retrieval_chunks.append(f"Type hint for {parameter} is unavailable.")
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameter_types_pairs))
        else:
            class_parameter_types_pairs = [
                (
                    f"Name type hint for {parameter}.",
                    f"{parameter} has '{parameter_annotation}' as type hint.",
                ),
                (
                    f"What is the type hint for {parameter}?",
                    f"The type hint for {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"Could you tell me the type hint for {parameter}?",
                    f"Sure, the type hint for {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"I need to know the type hint for {parameter}.",
                    f"The type hint for {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"Identify the type hint for {parameter}.",
                    f"The type hint for {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"Can you specify the type hint for {parameter}?",
                    f"Yes, the type hint for {parameter} is '{parameter_annotation}'.",
                ),
            ]
            class_member_retrieval_chunks.append(
                f"{parameter} is annotated as '{parameter_annotation}' type."
            )
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameter_types_pairs))

        if not (parameter_summary := class_parameter.parameter_summary):
            class_parameter_summary_pairs = [
                (
                    f"What does {parameter} do?",
                    f"Docstring of {class_member} does not describe '{parameter_name}'.",
                ),
                (
                    f"Can you explain the role of {parameter}?",
                    f"The docstring of {class_member} does not provide any information about"
                    f" '{parameter_name}'.",
                ),
                (
                    f"I'm trying to understand what {parameter} does. Can you help?",
                    f"Unfortunately, the docstring of {class_member} does not mention anything"
                    f" about '{parameter_name}'.",
                ),
                (
                    f"What is the function of {parameter}?",
                    f"There is no description of '{parameter_name}' in the docstring of"
                    f" {class_member}.",
                ),
                (
                    f"Could you tell me what '{parameter_name}' does in {class_member}?",
                    f"The docstring of {class_member} does not contain any details about"
                    f" '{parameter_name}'.",
                ),
                (
                    f"I'm curious about the purpose of {parameter}. Can you enlighten me?",
                    f"I'm sorry, but the docstring of {class_member} does not discuss"
                    f" '{parameter_name}'.",
                ),
            ]
            class_member_retrieval_chunks.append(
                f"{parameter} lacks any documentation in the docstring."
            )
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameter_summary_pairs))
        else:
            class_parameter_summary_pairs = [
                (
                    f"What does {parameter} do?",
                    f"{class_member} documents role of '{parameter_name}' as follows:"
                    f" '{parameter_summary}'.",
                ),
                (
                    f"Can you explain the role of {parameter}?",
                    f"Sure, {class_member} describes '{parameter_name}' as follows:"
                    f" '{parameter_summary}'.",
                ),
                (
                    f"I'm curious about {parameter}. What does it do?",
                    f"In {class_member}, '{parameter_name}' is documented as follows:"
                    f" '{parameter_summary}'.",
                ),
                (
                    f"Could you tell me what {parameter} does?",
                    f"Of course, {parameter} is described as follows: '{parameter_summary}'.",
                ),
                (
                    f"What's the function of {parameter}?",
                    f"{class_member} describes the function of '{parameter_name}' as follows:"
                    f" '{parameter_summary}'.",
                ),
                (
                    f"I'd like to know the purpose of {parameter}.",
                    f"In {class_member}, the purpose of '{parameter_name}' is defined as follows:"
                    f" '{parameter_summary}'.",
                ),
            ]
            class_member_retrieval_chunks.append(
                f"As per docstring, role of {parameter} is: '{parameter_summary}'."
            )
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_parameter_summary_pairs))

    if not (class_methods := member_type_details.class_methods):
        class_method_names_pairs = [
            (
                f"List names of the public methods of {class_member}.",
                f"{class_member} does not have any public methods (not starting with '_').",
            ),
            (
                f"Can you provide the names of the public methods for {class_member}?",
                f"Unfortunately, {class_member} does not have any public methods.",
            ),
            (
                f"What are the public methods of {class_member}?",
                f"There are no public methods (not starting with '_') in {class_member}.",
            ),
            (
                f"I need to know the public methods of {class_member}. Can you list them?",
                f"I'm sorry, but {class_member} does not have any public methods.",
            ),
            (
                f"Could you list the public methods of {class_member}?",
                f"{class_member} does not contain any public methods (not starting with '_').",
            ),
            (
                f"Show me the public methods of {class_member}.",
                f"It appears that {class_member} does not have any public methods.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} has no public (without _ as the prefix) methods."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_method_names_pairs))
    else:
        class_methods_count = len(class_methods)
        class_methods_count_pairs = [
            (
                f"How many public methods does {class_member} have?",
                f"{class_member} has {class_methods_count} many public methods.",
            ),
            (
                f"What is the count of public methods in {class_member}?",
                f"The count of public methods in {class_member} is {class_methods_count}.",
            ),
            (
                f"Could you tell me the number of public methods in {class_member}?",
                f"{class_member} has {class_methods_count} public methods.",
            ),
            (
                f"Please provide the count of public methods for {class_member}.",
                f"The number of public methods in {class_member} is {class_methods_count}.",
            ),
            (
                f"Tell me the quantity of public methods present in {class_member}.",
                f"{class_member} has {class_methods_count} public methods.",
            ),
            (
                f"Would you mind letting me know how many public methods {class_member} contains?",
                f"{class_member} contains {class_methods_count} public methods.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} has {class_methods_count} many public methods."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_methods_count_pairs))

        class_public_methods = enumerate_array_elements(class_methods, attribute="method_name")
        class_method_names_pairs = [
            (
                f"List names of the public methods of {class_member}.",
                f"Here are the public methods of {class_member}: {class_public_methods}.",
            ),
            (
                f"Can you provide the names of the public methods for {class_member}?",
                f"Sure, the public methods of {class_member} that do not start with '_' are:"
                f" {class_public_methods}.",
            ),
            (
                f"What are the public methods of {class_member}?",
                f"The public methods of {class_member} (excluding those starting with '_') are:"
                f" {class_public_methods}.",
            ),
            (
                f"I need to know the public methods of {class_member}.",
                f"The public methods of {class_member} (those not starting with '_') are:"
                f" {class_public_methods}.",
            ),
            (
                f"Could you list the public methods of {class_member}?",
                f"Of course, the public methods of {class_member} (not beginning with '_') are:"
                f" {class_public_methods}.",
            ),
            (
                f"Please show me the public methods of {class_member}.",
                f"Here you go, the public methods of {class_member}"
                f" (excluding those with a prefix '_') are: {class_public_methods}.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} has the following public methods: {class_public_methods}"
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_method_names_pairs))

    for class_method in class_methods:
        method_name = class_method.method_name
        method = f"'{method_name}' method of {class_member}"

        if not (method_parameters := class_method.method_parameters):
            class_method_parameters_pairs = [
                (f"What arguments do {method} accept?", f"{method} does not take any parameters."),
                (
                    f"Can you tell me the parameters that {method} requires?",
                    f"The {method} does not require any parameters.",
                ),
                (
                    f"What are the inputs for the {method} in {class_member}?",
                    f"There are no inputs for the {method} in {class_member}.",
                ),
                (
                    f"Does the {method} need any arguments?",
                    f"No, {method} does not need any arguments.",
                ),
                (
                    f"What parameters should I pass to {method}?",
                    f"You don't need to pass any parameters to the {method}.",
                ),
                (
                    f"What are required arguments for {method}?",
                    f"{method} does not require any arguments.",
                ),
            ]
            class_member_retrieval_chunks.append(f"{method} takes no arguments.")
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_method_parameters_pairs))
        else:
            class_method_parameters = enumerate_array_elements(method_parameters)
            class_method_parameters_pairs = [
                (
                    f"What arguments do {method} accept?",
                    f"{method} takes the following parameters: {class_method_parameters}.",
                ),
                (
                    f"Can you tell me the parameters that {method} requires?",
                    f"Sure, {method} requires these parameters: {class_method_parameters}.",
                ),
                (
                    f"I need to know arguments for {method}.",
                    f"The {method} has these arguments: {class_method_parameters}.",
                ),
                (
                    f"What are the parameters for '{method}'?",
                    f"The parameters for {method} are: {class_method_parameters}.",
                ),
                (
                    f"Could you list the arguments that the {method} takes?",
                    f"Certainly, the {method} takes these arguments: {class_method_parameters}.",
                ),
            ]
            class_member_retrieval_chunks.append(
                f"{method} accepts following parameters: {class_method_parameters}"
            )
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_method_parameters_pairs))

        if not (method_summary := class_method.method_summary):
            class_method_summary_pairs = [
                (f"What does {method} do?", f"Docstring of {method} is missing."),
                (
                    f"Can you explain functionality of {method}?",
                    f"The docstring for {method} is not available.",
                ),
                (
                    f"I'm trying to understand what {method} does. Can you help?",
                    f"Unfortunately, the docstring for {method} is not provided.",
                ),
                (
                    f"Could you describe the role of {method}?",
                    f"There is no docstring available for {method}.",
                ),
                (
                    f"I'm not sure what {method} does. Can you clarify?",
                    f"The {method} lacks a docstring.",
                ),
                (f"What's the purpose of {method}?", f"The {method} doesn't have a docstring."),
            ]
            class_member_retrieval_chunks.append(f"Unfortunately, {method} is not documented.")
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_method_summary_pairs))
        else:
            class_method_summary_pairs = [
                (
                    f"What does {method} do?",
                    f"Based on method docstring, its role is to '{method_summary}'.",
                ),
                (
                    f"Can you explain the function of {method}?",
                    f"Sure, according to method docstring, it is designed to '{method_summary}'.",
                ),
                (
                    f"I'm curious about the {method}. What's its purpose?",
                    f"Well, if we look at the docstring of {method}, we can see that it's meant to"
                    f" '{method_summary}'.",
                ),
                (
                    f"Could you tell me what the {method} does?",
                    f"Of course, the docstring of {method} indicates that its function is to"
                    f" '{method_summary}'.",
                ),
                (
                    f"I'd like to understand role of {method}.",
                    f"Certainly, method docstring reveals that its job is to '{method_summary}'.",
                ),
                (
                    f"What's the functionality of the {method}?",
                    f"As per the method docstring, it's designed to '{method_summary}'.",
                ),
            ]
            class_member_retrieval_chunks.append(
                f"Based on docstring, {method} has the purpose of '{method_summary}'."
            )
            class_member_tuning_pairs.extend(allocate_tuning_pairs(class_method_summary_pairs))

    if not (class_attributes := member_type_details.class_attributes):
        class_attribute_names_pairs = [
            (
                f"Are there any public attributes of {class_member}?",
                f"{class_member} has no public attributes (not starting with '_').",
            ),
            (
                f"Does {class_member} have any public attributes?",
                f"No, {class_member} does not have any public attributes.",
            ),
            (
                f"Can you tell me if {class_member} has any public attributes?",
                f"{class_member} does not have any public attributes (not starting with '_').",
            ),
            (
                f"I'm looking for public attributes of {class_member}. Are there any?",
                f"There are no public attributes (not starting with '_') for {class_member}.",
            ),
            (
                f"Is it possible to find any public attributes in {class_member}?",
                f"It's not possible to find any public attributes in {class_member}.",
            ),
        ]
        class_member_retrieval_chunks.append(f"{class_member} has no public attributes.")
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_attribute_names_pairs))
    else:
        class_attributes_count = len(class_attributes)
        class_attributes_count_pairs = [
            (
                f"How many public attributes does {class_member} have?",
                f"{class_member} has {class_attributes_count} many public attributes.",
            ),
            (
                f"What is the count of public attributes in {class_member}?",
                f"The count of public attributes in {class_member} is {class_attributes_count}.",
            ),
            (
                f"Could you tell me the number of public attributes in {class_member}?",
                f"{class_member} has {class_attributes_count} public attributes.",
            ),
            (
                f"Please provide the count of public attributes for {class_member}.",
                f"Number of public attributes in {class_member} is {class_attributes_count}.",
            ),
            (
                f"Tell me the quantity of public attributes present in {class_member}.",
                f"{class_member} has {class_attributes_count} public attributes.",
            ),
            (
                f"Would you mind letting me know how many public attributes {class_member}"
                " contains?",
                f"{class_member} contains {class_attributes_count} public attributes.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} has {class_attributes_count} many public attributes."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_attributes_count_pairs))

        class_public_attributes = enumerate_array_elements(
            class_attributes, attribute="attribute_name"
        )
        class_attribute_names_pairs = [
            (
                f"Are there any public attributes of {class_member}?",
                f"These are the public attributes of {class_member}: {class_public_attributes}.",
            ),
            (
                f"Can you list the public attributes of {class_member}?",
                f"{class_member} has the following public attributes (not starting with '_'):"
                f" {class_public_attributes}.",
            ),
            (
                f"What are the public attributes of {class_member}?",
                f"The public attributes of {class_member} (those not starting with '_') are:"
                f" {class_public_attributes}.",
            ),
            (
                f"I need to know the public attributes of {class_member}.",
                f"Sure, the public attributes of {class_member} are: {class_public_attributes}.",
            ),
            (
                f"Could you tell me the public attributes of {class_member}?",
                f"Of course, public attributes of {class_member} (not starting with '_') are:"
                f" {class_public_attributes}.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} has following public attributes: {class_public_attributes}"
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_attribute_names_pairs))

    if not (class_summary := member_type_details.class_summary):
        class_summary_pairs = [
            (
                f"What does {class_member} do in short?",
                f"Docstring of {class_member} lacks a summary of its objective.",
            ),
            (
                f"Can you briefly explain the function of {class_member}?",
                f"Docstring of {class_member} doesn't provide a concise summary of its purpose.",
            ),
            (
                f"Could you tell me what {class_member} is used for?",
                f"Unfortunately, the docstring of {class_member} doesn't contain"
                " a brief description of its function.",
            ),
            (
                f"I'm not sure what {class_member} does. Can you clarify?",
                f"The docstring of {class_member} doesn't succinctly explain its role.",
            ),
            (
                f"What's the purpose of {class_member}?",
                f"Docstring of {class_member} doesn't have any explanation of its objective.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"Unfortunately, {class_member} does not document its objective."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_summary_pairs))
    else:
        class_summary_pairs = [
            (
                f"What does {class_member} do in short?",
                f"Based on documentation, objective of {class_member} is to: '{class_summary}'.",
            ),
            (
                f"Can you briefly explain the function of {class_member}?",
                f"Sure, according to the documentation, {class_member} is designed to:"
                f" '{class_summary}'.",
            ),
            (
                f"I'm curious about {class_member}, what's its purpose?",
                f"Well, as per the documentation, {class_member} aims to: '{class_summary}'.",
            ),
            (
                f"Could you give me a quick rundown on what {class_member} does?",
                f"Absolutely, the documentation states that the role of {class_member} is to:"
                f" '{class_summary}'.",
            ),
            (
                f"What's the role of {class_member} in a nutshell?",
                f"The documentation indicates that the purpose of {class_member} is to:"
                f" '{class_summary}'.",
            ),
            (
                f"Can you summarise the function of {class_member}?",
                f"Of course, the documentation outlines that {class_member} is intended to:"
                f" '{class_summary}'.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"{class_member} documents its purpose as follows: '{class_summary}'."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_summary_pairs))

    if not (class_notes := member_type_details.class_notes):
        class_notes_pairs = [
            (
                f"Mention any specific details for {class_member} to be aware of.",
                f"Docstring of {class_member} does not note on specific details.",
            ),
            (
                f"What are the specific details to be aware of for {class_member}?",
                f"There are no specific details noted in the docstring of {class_member}.",
            ),
            (
                f"Could you tell me any specifics for {class_member} that I should be aware of?",
                f"The docstring of {class_member} doesn't highlight any details.",
            ),
            (
                f"Are there any specific details for {class_member} that I need to know?",
                f"No specific details are mentioned in the docstring of {class_member}.",
            ),
            (
                f"I need to know the specific details for {class_member}. Can you provide them?",
                f"Unfortunately, the docstring of {class_member} does not contain any details.",
            ),
            (
                f"Can you specify any details for {class_member} that I should be aware of?",
                f"The docstring of {class_member} does not specify any details to be aware of.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"Docstring of {class_member} has contains no specific implementation details."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_notes_pairs))
    else:
        class_notes_pairs = [
            (
                f"Mention any specific details for {class_member} to be aware of.",
                f"The {class_member} docstring highlights the following: '{class_notes}'.",
            ),
            (
                f"What are specifics that I should be aware of before using {class_member}?",
                f"The details you should know to use {class_member} are highlighted in docstring:"
                f" '{class_notes}'.",
            ),
            (
                f"Could you specify the details for {class_member} to take note of?",
                f"Sure, the docstring for {class_member} specifies the following details:"
                f" '{class_notes}'.",
            ),
            (
                f"Can you list the details for {class_member} to keep in mind?",
                f"Certainly, the docstring for {class_member} lists the following details:"
                f" '{class_notes}'.",
            ),
            (
                f"What should users of {class_member} be mindful of?",
                f"The docstring for {class_member} mentions the following points to be mindful of:"
                f" '{class_notes}'.",
            ),
            (
                f"What details does the user of {class_member} need to know?",
                f"User of {class_member} needs to know the following details: '{class_notes}'.",
            ),
        ]
        class_member_retrieval_chunks.append(
            f"In docstring, {class_member} specifies the following: '{class_notes}'."
        )
        class_member_tuning_pairs.extend(allocate_tuning_pairs(class_notes_pairs))

    class_member_dataset = Dataset(
        retrieval_chunks=class_member_retrieval_chunks[:2], tuning_pairs=class_member_tuning_pairs
    )

    return class_member_dataset, class_member_retrieval_chunks


@pydantic.validate_call(validate_return=True)
def generate_function_member_dataset(  # noqa: C901, PLR0912, PLR0915
    function_member: str, function_docstring: str, member_type_details: FunctionDetails
) -> tuple[Dataset, list[str]]:
    """Create relevant question and answers based on function member details.

    Parameters
    ----------
    function_member : str
        name of the function member
    function_docstring : str
        ``__doc__`` attribute of the function member, if any
    member_type_details : FunctionDetails
        details of the function member

    Returns
    -------
    Dataset
        all documents for retrieval and tuning for querying function member
    list[str]
        only retrieval documents
    """
    function_member_retrieval_chunks: list[str] = [
        f"{function_member} is a Python function.",
        f"{function_member} has following docstring: {function_docstring}.",
    ]
    function_member_tuning_pairs: list[tuple[str, str, SplitName]] = []

    if not (function_parameters := member_type_details.function_parameters):
        function_parameters_pairs = [
            (
                f"List various parameters of {function_member}.",
                f"{function_member} does not take any parameters.",
            ),
            (
                f"What are the parameters of {function_member}?",
                f"{function_member} has no parameters.",
            ),
            (
                f"Could you tell me the parameters that {function_member} takes?",
                f"{function_member} doesn't require any parameters.",
            ),
            (
                f"I need to know the parameters for {function_member}.",
                f"There are no parameters for {function_member}.",
            ),
            (
                f"Can you list the parameters for {function_member}?",
                f"Actually, {function_member} doesn't have any parameters.",
            ),
            (
                f"Please provide the parameters of {function_member}.",
                f"Sorry, but {function_member} does not have any parameters.",
            ),
        ]
        function_member_retrieval_chunks.append(f"{function_member} takes no parameters.")
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_parameters_pairs))
    else:
        function_parameter_names = enumerate_array_elements(
            function_parameters, attribute="parameter_details"
        )
        function_parameters_pairs = [
            (
                f"List various parameters of {function_member}.",
                f"Different parameters of {function_member} are as follows:"
                f" {function_parameter_names}.",
            ),
            (
                f"What are the different parameters of {function_member}?",
                f"{function_member} has the following parameters: {function_parameter_names}.",
            ),
            (
                f"Could you tell me the parameters of {function_member}?",
                f"Sure, the parameters of {function_member} are: {function_parameter_names}.",
            ),
            (
                f"I need to know the parameters of {function_member}.",
                f"The parameters of {function_member} are: {function_parameter_names}.",
            ),
            (
                f"Can you list the parameters for {function_member}?",
                f"Yes, the parameters for {function_member} are: {function_parameter_names}.",
            ),
            (
                f"Please provide the parameters of {function_member}.",
                f"Parameters of {function_member} are as follows: {function_parameter_names}.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} takes the following parameters: {function_parameter_names}"
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_parameters_pairs))

    for function_parameter in function_parameters:
        parameter_name = function_parameter.parameter_name
        parameter = f"'{parameter_name}' argument in {function_member}"

        if (parameter_default := function_parameter.parameter_default) is inspect._empty:
            function_parameter_defaults_pairs = [
                (f"Default value of {parameter}?", f"{parameter} does not have a default value."),
                (
                    f"What is the default value for {parameter}?",
                    f"The {parameter} does not come with a default value.",
                ),
                (
                    f"Could you tell me default value of {parameter}?",
                    f"Sure, the {parameter} does not possess a default value.",
                ),
                (
                    f"I'm curious about default value of {parameter}.",
                    f"In response to your curiosity, {parameter} is not assigned a default value.",
                ),
                (
                    f"I'd like to know the default value of {parameter}.",
                    f"To answer your query, {parameter} does not hold a default value.",
                ),
                (
                    f"Can you inform me about the default value of {parameter}?",
                    f"Certainly, {parameter} does not contain a default value.",
                ),
            ]
            function_member_retrieval_chunks.append(f"{parameter} has no default value.")
            function_member_tuning_pairs.extend(
                allocate_tuning_pairs(function_parameter_defaults_pairs)
            )
        else:
            function_parameter_defaults_pairs = [
                (
                    f"Default value of {parameter}?",
                    f"{parameter} has default value of {parameter_default}.",
                ),
                (
                    f"What is the default value for {parameter}?",
                    f"The default value for {parameter} is {parameter_default}.",
                ),
                (
                    f"Could you tell me default value of {parameter}?",
                    f"Sure, the default value of {parameter} is {parameter_default}.",
                ),
                (
                    f"I would like to know the default value of {parameter}.",
                    f"The {parameter} has a default value of {parameter_default}.",
                ),
                (
                    f"Can you inform me about the default value of {parameter}?",
                    f"Of course, the {parameter} defaults to {parameter_default}.",
                ),
                (
                    f"I'm interested in default value of {parameter}.",
                    f"The default value of the {parameter} is {parameter_default}.",
                ),
            ]
            function_member_retrieval_chunks.append(
                f"{parameter} has the default value of {parameter_default}."
            )
            function_member_tuning_pairs.extend(
                allocate_tuning_pairs(function_parameter_defaults_pairs)
            )

        if (parameter_annotation := function_parameter.parameter_annotation) is inspect._empty:
            function_parameter_types_pairs = [
                (
                    f"What is type annotation of {parameter}?",
                    f"{parameter} does not have a type annotation.",
                ),
                (
                    f"Can you tell me type annotation of {parameter}?",
                    f"The {parameter} does not have a type annotation.",
                ),
                (
                    f"I'm curious about the type annotation of {parameter}."
                    " Can you provide some information?",
                    f"Sure, the {parameter} does not have a type annotation.",
                ),
                (
                    f"Do you have any information on the type annotation of {parameter}?",
                    f"Yes, the {parameter} does not have a type annotation.",
                ),
                (
                    f"Could you inform me about the type annotation of {parameter}?",
                    f"Certainly, {parameter} does not have a type annotation.",
                ),
                (
                    f"I'd like to know the type annotation of {parameter}.",
                    f"The {parameter} you're asking about does not have a type annotation.",
                ),
            ]
            function_member_retrieval_chunks.append(
                f"Unfortunately, type hint for {parameter} is missing."
            )
            function_member_tuning_pairs.extend(
                allocate_tuning_pairs(function_parameter_types_pairs)
            )
        else:
            function_parameter_types_pairs = [
                (
                    f"What is type annotation of {parameter}?",
                    f"Type annotation of {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"Can you tell me type annotation of {parameter}?",
                    f"Sure, the type annotation of {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"I'm curious about the type annotation of {parameter}. What is it?",
                    f"The type annotation of {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"Do you know type annotation of {parameter}?",
                    f"Yes, the type annotation of {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"Could you inform me about the type annotation of {parameter}?",
                    f"Of course, the type annotation of {parameter} is '{parameter_annotation}'.",
                ),
                (
                    f"What's the type annotation for {parameter}?",
                    f"The type annotation for {parameter} is '{parameter_annotation}'.",
                ),
            ]
            function_member_retrieval_chunks.append(
                f"{parameter} has '{parameter_annotation}' as type annotation."
            )
            function_member_tuning_pairs.extend(
                allocate_tuning_pairs(function_parameter_types_pairs)
            )

        if not (parameter_summary := function_parameter.parameter_summary):
            function_parameter_summary_pairs = [
                (
                    f"What is {parameter} for?",
                    f"Docstring of {function_member} lacks a description for '{parameter_name}'.",
                ),
                (
                    f"Can you explain the purpose of {parameter}?",
                    f"The docstring of {function_member} doesn't provide a description.",
                ),
                (
                    f"I'm not sure what {parameter} does. Can you help?",
                    f"Unfortunately, the docstring of {function_member} doesn't include"
                    " a description.",
                ),
                (
                    f"Could you clarify the role of {parameter}?",
                    f"The description is missing in the docstring of {function_member}.",
                ),
                (
                    f"I'm confused about the {parameter}. What does it do?",
                    f"The docstring of {function_member} doesn't contain a description.",
                ),
                (
                    f"What does {parameter} do?",
                    f"There's no description in the docstring of {function_member}.",
                ),
            ]
            function_member_retrieval_chunks.append(
                f"{parameter} is not documented in the docstring."
            )
            function_member_tuning_pairs.extend(
                allocate_tuning_pairs(function_parameter_summary_pairs)
            )
        else:
            function_parameter_summary_pairs = [
                (
                    f"What is {parameter} for?",
                    f"Based on {function_member} docstring, its role is '{parameter_summary}'.",
                ),
                (
                    f"Can you explain the role of {parameter}?",
                    f"Sure, according to the docstring of {function_member},"
                    f" '{parameter_name}' is used for '{parameter_summary}'.",
                ),
                (
                    f"I'm curious about the {parameter}. What does it do?",
                    f"Well, if you look at the docstring of {function_member}, you'll see that"
                    f" '{parameter_name}' is responsible for '{parameter_summary}'.",
                ),
                (
                    f"Could you tell me the purpose of {parameter}?",
                    f"Of course, the docstring of {function_member} indicates that"
                    f" '{parameter_name}' serves the purpose of '{parameter_summary}'.",
                ),
                (
                    f"What's the function of {parameter}?",
                    f"As per the docstring of {function_member}, '{parameter_name}' functions as:"
                    f" '{parameter_summary}'.",
                ),
                (
                    f"I'd like to know what '{parameter_name}' does in {function_member}.",
                    f"Sure thing, the docstring of {function_member} states that"
                    f" '{parameter_name}' does '{parameter_summary}'.",
                ),
            ]
            function_member_retrieval_chunks.append(
                f"In the docstring, {parameter} is described as '{parameter_summary}'."
            )
            function_member_tuning_pairs.extend(
                allocate_tuning_pairs(function_parameter_summary_pairs)
            )

    if (
        returns_annotation := member_type_details.function_returns.returns_annotation
    ) is inspect._empty:
        function_return_type_pairs = [
            (
                f"What is the return type annotation of {function_member}?",
                f"{function_member} lacks a return type annotation. It may still return though.",
            ),
            (
                f"Can you tell me the return type annotation of {function_member}?",
                f"The function {function_member} does not have a return type annotation."
                " However, it may still return.",
            ),
            (
                f"I'm curious about return type annotation of {function_member}. What is it?",
                f"Well, {function_member} doesn't have a return type annotation."
                " But, it could still return.",
            ),
            (
                f"Do you know the return type annotation of {function_member}?",
                f"Actually, {function_member} doesn't come with a return type annotation."
                " It's possible that it still returns though.",
            ),
            (
                f"Could you inform me about the return type annotation of {function_member}?",
                f"Sure, {function_member} is missing a return type annotation."
                " It might still return though.",
            ),
            (
                f"What's the return type annotation for {function_member}?",
                f"It appears that {function_member} is without a return type annotation."
                " It may still have a return.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} has no return annotation, but its return can still be non-null."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_return_type_pairs))
    else:
        function_return_type_pairs = [
            (
                f"What is the return type annotation of {function_member}?",
                f"Return type annotation for {function_member} is '{returns_annotation}'.",
            ),
            (
                f"Can you tell me the return type annotation of {function_member}?",
                f"Sure, return type annotation for {function_member} is '{returns_annotation}'.",
            ),
            (
                f"I need to know the return type annotation of {function_member}.",
                f"The return type annotation for {function_member} is '{returns_annotation}'.",
            ),
            (
                f"Do you know the return type annotation of {function_member}?",
                f"Yes, return type annotation for {function_member} is '{returns_annotation}'.",
            ),
            (
                f"Could you inform me about the return type annotation of {function_member}?",
                f"Of course, the return type for {function_member} is '{returns_annotation}'.",
            ),
            (
                f"I'm curious about the return type annotation of {function_member}.",
                f"The return type annotation for {function_member} is '{returns_annotation}'.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"Return of {function_member} is annotated as '{returns_annotation}'."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_return_type_pairs))

    if not (returns_summary := member_type_details.function_returns.returns_summary):
        function_return_summary_pairs = [
            (
                f"What does {function_member} return?",
                f"Docstring of {function_member} does not describe its return.",
            ),
            (
                f"Can you tell me what {function_member} returns?",
                f"Docstring of {function_member} doesn't provide information about its return.",
            ),
            (
                f"Do you know the return of {function_member}?",
                f"Unfortunately, docstring of {function_member} doesn't specify what it returns.",
            ),
            (
                f"I'm curious about what {function_member} returns. Can you help?",
                f"I'm sorry, but the docstring of {function_member} doesn't clarify its return.",
            ),
            (
                f"What's the return of {function_member}?",
                f"The return of {function_member} is not described in its docstring.",
            ),
            (
                f"Could you inform me about the return of {function_member}?",
                f"Regrettably, the docstring of {function_member} doesn't detail its return.",
            ),
        ]
        function_member_retrieval_chunks.append(f"{function_member} does not document its return.")
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_return_summary_pairs))
    else:
        function_return_summary_pairs = [
            (
                f"What does {function_member} return?",
                f"Based on {function_member} docstring, the return contains: '{returns_summary}'.",
            ),
            (
                f"Can you tell me what {function_member} returns?",
                f"Sure, as per docstring of {function_member}, it returns: '{returns_summary}'.",
            ),
            (
                f"I'm curious about what {function_member} returns. Can you help?",
                f"Absolutely! The docstring of {function_member} indicates that it returns:"
                f" '{returns_summary}'.",
            ),
            (
                f"Do you know what {function_member} returns?",
                f"Yes, the docstring of {function_member} states that it returns:"
                f" '{returns_summary}'.",
            ),
            (
                f"I'd like to know what {function_member} returns.",
                f"Of course, the docstring of {function_member} reveals that its return contains:"
                f" '{returns_summary}'.",
            ),
            (
                f"Could you inform me about the return of {function_member}?",
                f"Certainly, the docstring of {function_member} specifies that it returns:"
                f" '{returns_summary}'.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"Based on docstring, return of {function_member} is as follows: '{returns_summary}'."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_return_summary_pairs))

    if not (function_summary := member_type_details.function_summary):
        function_summary_pairs = [
            (
                f"Summarise role of {function_member} in short.",
                f"{function_member} docstring lacks a summary of its objective.",
            ),
            (
                f"Can you briefly explain the role of {function_member}?",
                f"The docstring of {function_member} doesn't provide its purpose.",
            ),
            (
                f"What is the purpose of {function_member} as per its docstring?",
                f"The docstring of {function_member} doesn't clearly state its purpose.",
            ),
            (
                f"Could you provide a summary of objective of {function_member}?",
                f"The objective of {function_member} is not summarised in its docstring.",
            ),
            (
                f"What does {function_member} do according to its docstring?",
                f"According to its docstring, role of {function_member} is not summarised.",
            ),
        ]
        function_member_retrieval_chunks.append(f"Documentation for {function_member} is missing.")
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_summary_pairs))
    else:
        function_summary_pairs = [
            (
                f"Summarise role of {function_member} in short.",
                f"Based on docstring, objective of {function_member} is to: '{function_summary}'.",
            ),
            (
                f"Can you briefly explain the role of {function_member}?",
                f"Sure, according to the docstring, the purpose of {function_member} is:"
                f" '{function_summary}'.",
            ),
            (
                f"What does {function_member} do, in a nutshell?",
                f"In a nutshell, {function_member} is designed to: '{function_summary}'.",
            ),
            (
                f"Could you provide a short summary of role of {function_member}?",
                f"Certainly, from docstring, {function_member} aims to: '{function_summary}'.",
            ),
            (
                f"I need a brief explanation of what {function_member} does.",
                f"Of course, {function_member} is intended to: '{function_summary}'.",
            ),
            (
                f"In brief, what is the role of {function_member}?",
                f"Briefly, the role of {function_member} is to: '{function_summary}',"
                " according to the docstring.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} documents itself as follows: '{function_summary}'."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_summary_pairs))

    if not (function_raises := member_type_details.function_raises):
        function_raise_types_pairs = [
            (
                f"Does {function_member} raise any specific exception?",
                f"Docstring of {function_member} does not mention any specific exceptions.",
            ),
            (
                f"Are there any specific exceptions that {function_member} raises?",
                f"No specific exceptions are mentioned in the docstring of {function_member}.",
            ),
            (
                f"Can you tell me if {function_member} raises any specific exceptions?",
                f"According to docstring, {function_member} does not raise exceptions.",
            ),
            (
                f"I want to know if {function_member} raises any specific exceptions."
                " Can you confirm?",
                f"I can confirm that docstring of {function_member} does not mention exceptions.",
            ),
            (
                f"Could {function_member} possibly raise any specific exceptions?",
                f"The docstring of {function_member} does not indicate that"
                " it raises any specific exceptions.",
            ),
            (
                f"Is it possible for {function_member} to raise any specific exceptions?",
                f"The docstring of {function_member} does not suggest that"
                " it raises any specific exceptions.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} does not document any specific exceptions in the docstring."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_raise_types_pairs))
    else:
        function_raise_types = enumerate_array_elements(
            function_raises, attribute="raises_details"
        )
        function_raise_types_pairs = [
            (
                f"Does {function_member} raise any specific exception?",
                f"Based on docstring of {function_member}, it can raise the following:"
                f" {function_raise_types}.",
            ),
            (
                f"Can you tell me if {function_member} raises any specific exceptions?",
                f"Yes, according to docstring of {function_member}, it can raise these exceptions:"
                f" {function_raise_types}.",
            ),
            (
                f"What exceptions, if any, does {function_member} raise?",
                f"{function_member} can raise these exceptions as per its docstring:"
                f" {function_raise_types}.",
            ),
            (
                f"I need to know if {function_member} throws any specific exceptions."
                " Can you help?",
                f"Sure, {function_member} can throw following exceptions according to docstring:"
                f" {function_raise_types}.",
            ),
            (
                f"Could you inform me about any specific exceptions that"
                f" {function_member} might raise?",
                f"Certainly, the docstring of {function_member} indicates that"
                f" it can raise these exceptions: {function_raise_types}.",
            ),
            (
                f"I'm curious about the exceptions that {function_member} might throw."
                " Do you have any information?",
                f"Yes, the docstring of {function_member} suggests that"
                f" it can throw the following exceptions: {function_raise_types}.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"From docstring, {function_member} can raise the following: {function_raise_types}"
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_raise_types_pairs))

    if not (function_warns := member_type_details.function_warns):
        function_warn_types_pairs = [
            (
                f"Does {function_member} throw any specific warnings?",
                f"Docstring of {function_member} lacks any mention of specific warnings.",
            ),
            (
                f"Are there any specific warnings that {function_member} throws?",
                f"There are no specific warnings mentioned in docstring of {function_member}.",
            ),
            (
                f"Can you tell me if {function_member} throws any specific warnings?",
                f"According to the docstring of {function_member},"
                " it doesn't throw any specific warnings.",
            ),
            (
                f"I want to know if {function_member} throws any specific warnings."
                " Can you help?",
                f"Sure, I checked the docstring of {function_member} and"
                " found no mention of specific warnings.",
            ),
            (
                f"Could you check if {function_member} throws any specific warnings?",
                f"I've checked the docstring of {function_member} and"
                " it doesn't mention any specific warnings.",
            ),
            (
                f"Is it possible that {function_member} throws any specific warnings?",
                f"Based on the docstring of {function_member},"
                " it doesn't seem to throw any specific warnings.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"Mention of any warnings is missing in docstring of {function_member}."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_warn_types_pairs))
    else:
        function_warn_types = enumerate_array_elements(function_warns, attribute="warns_details")
        function_warn_types_pairs = [
            (
                f"Does {function_member} throw any specific warnings?",
                f"Based on the docstring, {function_member} can throw the following warnings:"
                f" {function_warn_types}.",
            ),
            (
                f"Can you tell me if {function_member} throws any specific warnings?",
                f"Yes, according to docstring, {function_member} may throw these warnings:"
                f" {function_warn_types}.",
            ),
            (
                f"I'm curious, does {function_member} generate any particular warnings?",
                f"Indeed, docstring indicates that {function_member} can generate these warnings:"
                f" {function_warn_types}.",
            ),
            (
                f"What specific warnings, if any, does {function_member} throw?",
                f"{function_member} throws the following warnings as per the docstring:"
                f" {function_warn_types}.",
            ),
            (
                f"Could {function_member} possibly throw any specific warnings?",
                f"Yes, it could. Docstring of {function_member} mentions these specific warnings:"
                f" {function_warn_types}.",
            ),
            (
                f"Are there any specific warnings that {function_member} throws?",
                f"Yes, there are. The docstring for {function_member} lists following warnings:"
                f" {function_warn_types}.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} documents the following warnings: {function_warn_types}"
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_warn_types_pairs))

    if not (function_notes := member_type_details.function_notes):
        function_notes_pairs = [
            (
                f"Is there any specific details for {function_member} to be aware of?",
                f"Docstring of {function_member} lacks any notes on specific details.",
            ),
            (
                f"Are there any particular details I should know about {function_member}?",
                f"There are no specific details noted in the docstring of {function_member}.",
            ),
            (
                f"What should I be aware of when using {function_member}?",
                f"The docstring of {function_member} does not contain any details to be aware of.",
            ),
            (
                f"Could you tell me if there are any specific details for {function_member}?",
                f"No specific details are mentioned in the docstring of {function_member}.",
            ),
            (
                f"I'm curious if there are any specific details about {function_member}?",
                f"The docstring of {function_member} does not provide any specific details.",
            ),
            (
                f"Do I need to be aware of any specific details for {function_member}?",
                f"The docstring of {function_member} does not include any specific details.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} has no specific notes in the docstring."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_notes_pairs))
    else:
        function_notes_pairs = [
            (
                f"Is there any specific details for {function_member} to be aware of?",
                f"Docstring of {function_member} highlights the following: '{function_notes}'.",
            ),
            (
                f"What should I know about {function_member}?",
                "You should be aware that docstring includes the following details:"
                f" '{function_notes}'.",
            ),
            (
                f"Could you provide some details about {function_member}?",
                f"Sure, the docstring of {function_member} provides the following information:"
                f" '{function_notes}'.",
            ),
            (
                f"What are the important details of {function_member}?",
                f"The important details of {function_member} are highlighted in its docstring:"
                f" '{function_notes}'.",
            ),
            (
                f"Can you tell me more about {function_member}?",
                f"Of course, the docstring of {function_member} contains the following details:"
                f" '{function_notes}'.",
            ),
            (
                f"I need information about {function_member}.",
                f"The docstring of {function_member} contains the following information:"
                f" '{function_notes}'.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"Docstring for {function_member} has following notes: '{function_notes}'."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_notes_pairs))

    if not (function_references := member_type_details.function_references):
        function_references_pairs = [
            (
                f"Is there any reference for {function_member}?",
                f"Documentation for {function_member} contains no references.",
            ),
            (
                f"Can I find any references in the documentation for {function_member}?",
                f"No, the documentation for {function_member} does not contain any references.",
            ),
            (
                f"Does the documentation for {function_member} include any references?",
                f"No, there are no references in the documentation for {function_member}.",
            ),
            (
                f"Are there references available in the {function_member} documentation?",
                f"No, the {function_member} documentation does not include any references.",
            ),
            (
                f"I'm looking for references in {function_member} documentation. Are there any?",
                f"Unfortunately, the documentation for {function_member} contains no references.",
            ),
            (
                f"Could you tell me if there are any references for {function_member}?",
                f"I'm sorry, but documentation for {function_member} lacks any references.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} documents no references in its docstring."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_references_pairs))
    else:
        function_references_pairs = [
            (
                f"Is there any reference for {function_member}?",
                f"The docstring links the following: '{function_references}'.",
            ),
            (
                f"Can you provide a reference for {function_member}?",
                f"Sure, the docstring provides the following reference: '{function_references}'.",
            ),
            (
                f"Where can I find a reference for {function_member}?",
                f"You can find it in the docstring, which links to: '{function_references}'.",
            ),
            (
                f"Could you point me to the reference for {function_member}?",
                f"Of course, the docstring points to these reference: '{function_references}'.",
            ),
            (
                f"I'm looking for a reference for {function_member}. Can you help?",
                f"Absolutely, the docstring links to this reference: '{function_references}'.",
            ),
            (
                f"What's the reference for {function_member}?",
                f"The reference for that is in the docstring: '{function_references}'.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"{function_member} list the following references: {function_references}"
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_references_pairs))

    if not (function_examples := member_type_details.function_examples):
        function_examples_pairs = [
            (
                f"Is there any example for {function_member}?",
                f"Docstring for {function_member} lacks any examples.",
            ),
            (
                f"Can I find an example for {function_member} in the docstring?",
                f"Unfortunately, docstring for {function_member} does not contain any examples.",
            ),
            (
                f"Does the docstring for {function_member} include any examples?",
                f"No, the docstring for {function_member} does not include any examples.",
            ),
            (
                f"I'm looking for an example of {function_member} in docstring, is there one?",
                f"I'm sorry, but docstring for {function_member} does not provide any examples.",
            ),
            (
                f"Are there any examples provided in the docstring for {function_member}?",
                f"No examples are provided in the docstring for {function_member}.",
            ),
            (
                f"Could you tell me if there's an example for {function_member} in docstring?",
                f"I regret to inform you that {function_member} documents no examples.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"Documentation of {function_member} lacks any examples."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_examples_pairs))
    else:
        function_examples_pairs = [
            (
                f"Is there any example for {function_member}?",
                f"Documentation of {function_member} contains these examples:"
                f" '{function_examples}'.",
            ),
            (
                f"Can you provide an example of {function_member}?",
                f"Sure, you can find examples of {function_member} in its documentation:"
                f" '{function_examples}'.",
            ),
            (
                f"I'm looking for examples of {function_member}, can you help?",
                f"Absolutely, examples for {function_member} are available in its documentation:"
                f" '{function_examples}'.",
            ),
            (
                f"Where can I find examples for {function_member}?",
                f"You can find examples for {function_member} in its documentation:"
                f" '{function_examples}'.",
            ),
            (
                f"Could you show me some examples of {function_member}?",
                f"Of course, the documentation of {function_member} includes these examples:"
                f" '{function_examples}'.",
            ),
            (
                f"I need examples for {function_member}, where can I find them?",
                f"You can find examples for {function_member} in its documentation:"
                f" '{function_examples}'.",
            ),
        ]
        function_member_retrieval_chunks.append(
            f"Docstring of {function_member} contains following examples: '{function_examples}'."
        )
        function_member_tuning_pairs.extend(allocate_tuning_pairs(function_examples_pairs))

    function_member_dataset = Dataset(
        retrieval_chunks=function_member_retrieval_chunks[:2],
        tuning_pairs=function_member_tuning_pairs,
    )

    return function_member_dataset, function_member_retrieval_chunks


@pydantic.validate_call(validate_return=True)
def generate_member_dataset(member_details: MemberDetails) -> tuple[Dataset, ...]:
    """Create a dataset for a member.

    Parameters
    ----------
    member_details : MemberDetails
        all details of the member

    Returns
    -------
    tuple[Dataset, ...]
        all documents for retrieval and tuning pairs for querying member documentation

    Raises
    ------
    ValueError
        if the member type is not supported

    Notes
    -----
    * There will be a single return if member type is not enum, class or function.
    * Otherwise, there will be two returns, one for the member and one for the member type.
    """
    member_name = member_details.member_name
    member_full_name = member_details.member_qualified_name
    member = f"'{member_name}' object"

    member_retrieval_chunks: list[str] = []
    member_tuning_pairs: list[tuple[str, str, SplitName]] = []

    module_parent_pairs = [
        (
            f"What is the parent module of {member}?",
            f"'{member_details.member_module}' is the name of its parent module.",
        ),
        (
            f"Can you tell me the parent module of {member}?",
            f"Sure, the parent module of {member} is '{member_details.member_module}'.",
        ),
        (
            f"I'm trying to find the parent module of {member}, can you help?",
            f"Of course, parent module of {member} is '{member_details.member_module}'.",
        ),
        (
            f"Do you know the parent module of {member}?",
            f"Yes, the parent module of {member} is '{member_details.member_module}'.",
        ),
        (
            f"I need to know the parent module of {member}, can you provide that?",
            f"Absolutely, parent module of {member} is '{member_details.member_module}'.",
        ),
        (
            f"Could you inform me about the parent module of {member}?",
            f"Certainly, '{member_details.member_module}' is parent module of {member}.",
        ),
    ]
    member_retrieval_chunks.append(
        f"{member} is part of parent module {member_details.member_module}."
    )
    member_tuning_pairs.extend(allocate_tuning_pairs(module_parent_pairs))

    member_full_name_pairs = [
        (
            f"What is the full name of {member}?",
            f"'{member_full_name}' is its fully qualified name.",
        ),
        (
            f"Can you tell me the full name of the {member}?",
            f"Sure, the fully qualified name of {member} is '{member_full_name}'.",
        ),
        (
            f"I need to know the full name of {member}. Can you help?",
            f"Of course, the full name of {member} is '{member_full_name}'.",
        ),
        (
            f"What's the fully qualified name for the {member}?",
            f"The fully qualified name for {member} is '{member_full_name}'.",
        ),
        (
            f"Could you provide the full name of the {member}?",
            f"Certainly, the full name of the {member} is '{member_full_name}'.",
        ),
        (
            f"I'm looking for the full name of {member}. What is it?",
            f"The full name of {member} is '{member_full_name}'.",
        ),
    ]
    member_retrieval_chunks.append(f"Full name of {member} is '{member_full_name}'.")
    member_tuning_pairs.extend(allocate_tuning_pairs(member_full_name_pairs))

    member_hierarchy = enumerate_array_elements(member_details.member_hierarchy)
    member_hierarchy_pairs = [
        (
            f"What is the hierarchy of {member}?",
            f"The hierarchy of {member} is as follows: {member_hierarchy}.",
        ),
        (
            f"Can you explain the hierarchy of the {member}?",
            f"Sure, the hierarchy of the {member} is: {member_hierarchy}.",
        ),
        (
            f"Could you tell me the hierarchy of {member}?",
            f"Of course, the hierarchy of {member} is: {member_hierarchy}.",
        ),
        (
            f"I would like to know the hierarchy of {member}. Can you provide that?",
            f"Absolutely, the hierarchy of {member} is: {member_hierarchy}.",
        ),
        (
            f"Please provide the hierarchy of {member}.",
            f"The hierarchy of {member} is: {member_hierarchy}.",
        ),
        (
            f"I'm interested in the hierarchy of {member}. Could you share it?",
            f"Sure, the hierarchy of {member} is: {member_hierarchy}.",
        ),
    ]
    member_retrieval_chunks.append(f"Hierarchy of {member} is as follows: {member_hierarchy}.")
    member_tuning_pairs.extend(allocate_tuning_pairs(member_hierarchy_pairs))

    if not (member_docstring := member_details.member_docstring):
        member_documentation_pairs = [
            (
                f"What is the documentation of {member}?",
                f"{member} does not have any documentation.",
            ),
            (
                f"Can you provide the documentation for the {member}?",
                f"Sorry, the {member} does not have any documentation.",
            ),
            (
                f"Is there any documentation available for the {member}?",
                f"No, there is no documentation available for the {member}.",
            ),
            (
                f"Could you show me the documentation of the {member}?",
                f"Unfortunately, the {member} does not have any documentation.",
            ),
            (
                f"I'm looking for the documentation of {member}. Can you help?",
                f"I'm sorry, but the {member} does not have any documentation.",
            ),
        ]
        member_retrieval_chunks.append(
            f"Unfortunately, {member} currently does not have any documentation."
        )
        member_tuning_pairs.extend(allocate_tuning_pairs(member_documentation_pairs))
    else:
        member_documentation_pairs = [
            (f"What does {member} do?", f"Its documentation is as follows: '{member_docstring}'."),
            (
                f"Can you explain the function of the {member}?",
                f"Sure, here is its documentation: '{member_docstring}'.",
            ),
            (
                f"I'm not sure what {member} does. Can you clarify?",
                f"Of course, here's its documentation for clarification: '{member_docstring}'.",
            ),
            (
                f"Could you tell me about the {member}?",
                f"Certainly, its documentation is: '{member_docstring}'.",
            ),
            (
                f"I need information on the {member}.",
                f"Here's the documentation you need: '{member_docstring}'.",
            ),
            (
                f"What's the purpose of the {member}?",
                f"The purpose is described in its documentation: '{member_docstring}'.",
            ),
        ]
        member_retrieval_chunks.append(
            f"The following is the documentation of {member}: '{member_docstring}'."
        )
        member_tuning_pairs.extend(allocate_tuning_pairs(member_documentation_pairs))

    if (member_type_details := member_details.member_type_details) is not None:
        member_type = member_type_details.member_type

        member_type_pairs = [
            (f"What is the type of {member}?", f"{member} is of '{member_type.value}' type."),
            (
                f"Can you tell me the type of the {member}?",
                f"Sure, the {member} is of '{member_type.value}' type.",
            ),
            (
                f"I would like to know the type of {member}. Can you help?",
                f"Absolutely, the {member} is of '{member_type.value}' type.",
            ),
            (
                f"Do you know the type of {member}?",
                f"Yes, the {member} is of '{member_type.value}' type.",
            ),
            (
                f"Could you inform me about the type of {member}?",
                f"Of course, the {member} is of '{member_type.value}' type.",
            ),
            (
                f"I'm curious about type of {member}. Can you provide some information?",
                f"Certainly, the {member} is of '{member_type.value}' type.",
            ),
        ]
        member_retrieval_chunks.insert(-1, f"'{member_name}' is a Python {member_type.value}.")
        member_tuning_pairs.extend(allocate_tuning_pairs(member_type_pairs))

    if member_type_details is None:
        member_retrieval_chunks.insert(0, f"'{member_name}' is a Python object.")

        member_dataset = Dataset(
            retrieval_chunks=member_retrieval_chunks, tuning_pairs=member_tuning_pairs
        )

        return (member_dataset,)

    match member_type:
        case MemberType.ENUM:
            member_type_dataset, member_type_retrieval_chunks = generate_enum_member_dataset(
                f"'{member_name}' enum", member_docstring, member_type_details
            )
        case MemberType.CLASS:
            member_type_dataset, member_type_retrieval_chunks = generate_class_member_dataset(
                f"'{member_name}' class", member_docstring, member_type_details
            )
        case MemberType.FUNCTION:
            member_type_dataset, member_type_retrieval_chunks = generate_function_member_dataset(
                f"'{member_name}' function", member_docstring, member_type_details
            )
        case _:
            LOGGER.critical(f"Received unsupported {member_type=}")

            raise ValueError("Unexpected member type: supports 'enum', 'class', 'function'")

    member_dataset = Dataset(
        retrieval_chunks=member_retrieval_chunks + member_type_retrieval_chunks,
        tuning_pairs=member_tuning_pairs,
    )

    return (member_dataset, member_type_dataset)


__all__ = [
    "enumerate_array_elements",
    "generate_class_member_dataset",
    "generate_enum_member_dataset",
    "generate_function_member_dataset",
    "generate_member_dataset",
    "generate_module_dataset",
    "generate_package_dataset",
]
