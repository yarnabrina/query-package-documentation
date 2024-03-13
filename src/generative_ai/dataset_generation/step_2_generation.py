"""Define functionalities to generate retrieval and tuning sources."""

import inspect
import itertools
import logging
import random

import pydantic

from .utils_generation import (
    ClassDetails,
    Dataset,
    Document,
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

EMPTY_PARAMETER = inspect.Parameter.empty
EMPTY_SIGNATURE = inspect.Signature.empty


@pydantic.validate_call(validate_return=True)
def allocate_tuning_triplets(
    context: str,
    questions: list[str],
    answers: list[str],
    split_proportions: SplitProportions = DEFAULT_SPLIT_PROPORTIONS,
) -> list[Document]:
    """Allocate tuning triplets to different splits.

    Parameters
    ----------
    context : str
        source of information
    questions : list[str]
        queries from ``context``
    answers : list[str]
        responses based on ``context``
    split_proportions : SplitProportions, optional
        chance of a pair to be allocated to different splits, by default DEFAULT_SPLIT_PROPORTIONS

    Returns
    -------
    list[Document]
        updated tuning triplets with split allocation
    """
    question_answer_pairs = list(itertools.product(questions, answers))

    allocations = random.choices(  # noqa: S311
        [SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST],
        weights=[
            split_proportions.train_proportion,
            split_proportions.validation_proportion,
            split_proportions.test_proportion,
        ],
        k=len(question_answer_pairs),
    )

    return [
        Document(context=context, question=question, answer=answer, split=allocation)
        for (question, answer), allocation in zip(question_answer_pairs, allocations, strict=True)
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
    package_tuning_documents: list[Document] = []

    if (parent_package := package_contents.parent_package_name) is None:
        root_package_context = f"'{package_name}' is the root package."
        root_package_questions = [
            "What is the root package?",
            "Can you tell me what the root package is?",
            "I'm trying to find out the root package. Can you help?",
            "Do you know what the root package is?",
            "I'd like to know the root package.",
            "Could you identify the root package?",
        ]
        root_package_answers = [
            f"'{package_name}' is the root package.",
            f"The root package is '{package_name}'.",
            f"The root package you're asking about is '{package_name}'.",
        ]

        package_retrieval_chunks.append(root_package_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                root_package_context, root_package_questions, root_package_answers
            )
        )

        parent_package_context = f"'{package_name}' has no parent package."
        parent_package_questions = [
            f"Name parent package of '{package_name}'.",
            f"What is the parent package of '{package_name}'?",
            f"Can you tell me the parent package of '{package_name}'?",
            f"Could you identify the parent package of '{package_name}'?",
            f"I'm looking for the parent package of '{package_name}'. Can you help?",
            f"Do you know the parent package of '{package_name}'?",
        ]
        parent_package_answers = [
            f"Being the root package, '{package_name}' has no parent package.",
            f"The root package '{package_name}' does not have a parent package.",
            f"'{package_name}' is a root package and therefore it does not have a parent package.",
            f"As a root package, '{package_name}' does not possess a parent package.",
            f"'{package_name}' is a root package, so it doesn't have a parent package.",
            f"'{package_name}' is a root package and hence it doesn't have a parent package.",
        ]

        package_retrieval_chunks.append(parent_package_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                parent_package_context, parent_package_questions, parent_package_answers
            )
        )
    else:
        parent_package_context = f"'{package_name}' is part of parent package '{parent_package}'."
        parent_package_questions = [
            f"Name parent package of '{package_name}' sub-package.",
            f"What is the parent package of the '{package_name}' sub-package?",
            f"Could you tell me the parent package of '{package_name}'?",
            f"I need to know the parent package of '{package_name}'.",
            f"Identify the parent package for the '{package_name}' sub-package.",
            f"Can you name the parent package of the '{package_name}' sub-package?",
        ]
        parent_package_answers = [
            f"'{parent_package}' is the full name of its parent package.",
            f"The parent package of '{package_name}' is '{parent_package}'.",
            f"The parent package for '{package_name}' is identified as '{parent_package}'.",
        ]

        package_retrieval_chunks.append(parent_package_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                parent_package_context, parent_package_questions, parent_package_answers
            )
        )

        package_full_name_context = (
            f"Full name of '{package_name}' sub-package is '{package_full_name}'."
        )
        package_full_name_questions = [
            f"Tell the full name of '{package_name}' sub-package.",
            f"What is the fully qualified name of the '{package_name}' sub-package?",
            f"Could you provide the full name of the '{package_name}' sub-package?",
            f"I need the full name of the '{package_name}' sub-package. Can you tell me?",
            f"Can you inform me about the full name of the '{package_name}' sub-package?",
            f"Please, reveal the full name of the '{package_name}' sub-package.",
        ]
        package_full_name_answers = [
            f"'{package_full_name}' is the fully qualified name of '{package_name}'.",
            f"Fully qualified name of '{package_name}' sub-package is '{package_full_name}'.",
            f"The full name of '{package_name}' sub-package is '{package_full_name}'.",
        ]

        package_retrieval_chunks.append(package_full_name_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_full_name_context, package_full_name_questions, package_full_name_answers
            )
        )

        package_hierarchy = enumerate_array_elements(package_contents.package_hierarchy)

        package_hierarchy_context = f"Hierarchy of {package} is as follows: {package_hierarchy}."
        package_hierarchy_questions = [
            f"What is the hierarchy of {package}?",
            f"Can you explain the hierarchy of the {package}?",
            f"Could you describe the structure of the {package}?",
            f"I need to understand the hierarchy of {package}. Can you help?",
            f"Please provide the hierarchy of the {package}.",
            f"I'm interested in the structure of the {package}. What is it?",
        ]
        package_hierarchy_answers = [
            f"The hierarchy of {package} is as follows: {package_hierarchy}.",
            f"The hierarchy of {package} is: {package_hierarchy}.",
            f"The structure of {package} is: {package_hierarchy}.",
            f"The structure of {package} is as follows: {package_hierarchy}.",
        ]

        package_retrieval_chunks.append(package_hierarchy_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_hierarchy_context, package_hierarchy_questions, package_hierarchy_answers
            )
        )

    if not (children_sub_packages := package_contents.children_sub_packages_names):
        package_sub_package_context = f"{package} does not have any further sub-packages."
        package_sub_package_questions = [
            f"List the sub-packages of {package}.",
            f"What are the sub-packages of the {package}?",
            f"Could you tell me the sub-packages of {package}?",
            f"I need to know the sub-packages of {package}. Can you list them?",
            f"Can you provide a list of sub-packages for the {package}?",
            f"Identify the sub-packages of {package}.",
        ]
        package_sub_package_answers = [
            f"{package} does not have any further sub-packages.",
            f"The {package} does not contain any sub-packages.",
            f"The {package} doesn't have any sub-packages.",
            f"{package} doesn't include any sub-packages.",
            f"There are no sub-packages in the {package}.",
            f"No sub-packages are present in the {package}.",
        ]

        package_retrieval_chunks.append(package_sub_package_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_sub_package_context,
                package_sub_package_questions,
                package_sub_package_answers,
            )
        )
    else:
        children_sub_packages_count = len(children_sub_packages)

        children_sub_packages_count_context = (
            f"{package} has {children_sub_packages_count} many sub-packages."
        )
        children_sub_packages_count_questions = [
            f"How many sub-packages are there in {package}?",
            f"What is the count of sub-packages in {package}?",
            f"Could you tell me the number of sub-packages available in {package}?",
            f"Please provide the count of sub-packages for {package}.",
            f"Tell me the quantity of sub-packages present in {package}.",
            f"Would you mind letting me know how many sub-packages {package} contains?",
        ]
        children_sub_packages_count_answers = [
            f"{package} has {children_sub_packages_count} many sub-packages.",
            f"The count of sub-packages in {package} is {children_sub_packages_count}.",
            f"{package} has {children_sub_packages_count} sub-packages.",
            f"Number of sub-packages in {package} is {children_sub_packages_count}.",
            f"{package} has {children_sub_packages_count} sub-packages.",
            f"{package} contains {children_sub_packages_count} sub-packages.",
        ]

        package_retrieval_chunks.append(children_sub_packages_count_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                children_sub_packages_count_context,
                children_sub_packages_count_questions,
                children_sub_packages_count_answers,
            )
        )

        package_sub_packages = enumerate_array_elements(children_sub_packages)

        package_sub_package_context = (
            f"Sub-packages of {package} are as follows: {package_sub_packages}."
        )
        package_sub_package_questions = [
            f"List the sub-packages of {package}.",
            f"What are the sub-packages of the {package}?",
            f"Could you tell me the sub-packages of {package}?",
            f"I need to know the sub-packages of {package}. Can you list them?",
            f"Please provide the sub-packages of {package}.",
            f"Can you enumerate the sub-packages of {package}?",
        ]
        package_sub_package_answers = [
            f"Sub-packages of {package} are as follows: {package_sub_packages}.",
            f"The {package} has the following sub-packages: {package_sub_packages}.",
            f"The sub-packages of {package} are: {package_sub_packages}.",
        ]

        package_retrieval_chunks.append(package_sub_package_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_sub_package_context,
                package_sub_package_questions,
                package_sub_package_answers,
            )
        )

    if not (children_modules := package_contents.children_modules_names):
        package_module_context = f"{package} does not have any further modules."
        package_module_questions = [
            f"What are the modules of {package}?",
            f"Can you list the modules under the {package}?",
            f"Does the {package} contain any modules?",
            f"I'm looking for the modules of {package}. Can you help?",
            f"Tell me about the modules of {package}.",
            f"Are there any modules under the {package}?",
        ]
        package_module_answers = [
            f"{package} does not have any direct modules under itself.",
            f"There are no direct modules under the {package}.",
            f"No, the {package} does not contain any direct modules.",
            f"{package} does not have any direct modules.",
            f"The {package} does not have any direct modules.",
            f"There aren't any direct modules under the {package}.",
        ]

        package_retrieval_chunks.append(package_module_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_module_context, package_module_questions, package_module_answers
            )
        )
    else:
        children_modules_count = len(children_modules)

        children_modules_count_context = f"{package} has {children_modules_count} many modules."
        children_modules_count_questions = [
            f"How many modules are there in {package}?",
            f"What is the count of modules in {package}?",
            f"Could you tell me the number of modules available in {package}?",
            f"Please provide the count of modules for {package}.",
            f"Tell me the quantity of modules present in {package}.",
            f"Would you mind letting me know how many modules {package} contains?",
        ]
        children_modules_count_answers = [
            f"{package} has {children_modules_count} many modules.",
            f"The count of modules in {package} is {children_modules_count}.",
            f"{package} has {children_modules_count} modules.",
            f"The number of modules in {package} is {children_modules_count}.",
            f"{package} contains {children_modules_count} modules.",
        ]

        package_retrieval_chunks.append(children_modules_count_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                children_modules_count_context,
                children_modules_count_questions,
                children_modules_count_answers,
            )
        )

        package_modules = enumerate_array_elements(children_modules)

        package_module_context = f"Modules of {package} are as follows: {package_modules}."
        package_module_questions = [
            f"What are the modules of {package}?",
            f"Can you list the modules of the {package}?",
            f"I need to know the modules of the {package}.",
            f"Could you tell me what the modules of the {package} are?",
            f"I'm interested in the modules of the {package}.",
            f"What modules does the {package} contain?",
        ]
        package_module_answers = [
            f"Direct modules under {package} are as follows: {package_modules}.",
            f"The direct modules under {package} are: {package_modules}.",
            f"The modules you're looking for in {package} are: {package_modules}.",
            f"The modules under {package} are: {package_modules}.",
            f"The modules in {package} are: {package_modules}.",
            f"The {package} contains these modules: {package_modules}.",
        ]

        package_retrieval_chunks.append(package_module_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_module_context, package_module_questions, package_module_answers
            )
        )

    if not (package_summary := package_contents.package_summary):
        package_summary_context = (
            f"Unfortunately, {package} currently does not have any documentation."
        )
        package_summary_questions = [
            f"What does {package} do?",
            f"Can you tell me the functionality of the {package}?",
            f"I'm curious about what the {package} does. Can you enlighten me?",
            f"Could you explain the purpose of the {package}?",
            f"What's the role of the {package}?",
            f"What functionality does the {package} provide?",
        ]
        package_summary_answers = [
            f"{package} does not have any documentation.",
            f"The {package} provides no documentation.",
            f"The {package} does not come with any documentation.",
            f"The {package} lacks any form of documentation.",
            f"The {package} does not offer any documentation.",
            f"The {package} does not have any available documentation.",
        ]

        package_retrieval_chunks.append(package_summary_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_summary_context, package_summary_questions, package_summary_answers
            )
        )
    else:
        package_summary_context = (
            f"The following is the documentation of {package}: '{package_summary}'."
        )
        package_summary_questions = [
            f"What does {package} do?",
            f"Can you tell me about the {package}?",
            f"I'd like to know what the {package} does.",
            f"Could you explain the functionality of the {package}?",
            f"What's the purpose of the {package}?",
            f"I'm curious about the {package}, what does it do?",
        ]
        package_summary_answers = [
            f"Its documentation is as follows: '{package_summary}'.",
            f"Here is its documentation: '{package_summary}'.",
            f"Here's the documentation for it: '{package_summary}'.",
            f"The documentation states: '{package_summary}'.",
            f"The purpose is described in its documentation: '{package_summary}'.",
            f"Its documentation reads: '{package_summary}'.",
        ]

        package_retrieval_chunks.append(package_summary_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_summary_context, package_summary_questions, package_summary_answers
            )
        )

    if not (package_exports := package_contents.package_all_exports):
        package_members_context = (
            f"{package} does not export anything publicly using __all__ variable."
        )
        package_members_questions = [
            f"What are the public members of the {package}?",
            f"Can you list the public members of the {package}?",
            f"Are there any public members in the {package}?",
            f"I'm looking for public members of {package}. Can you help?",
            f"Could you tell me the public members of the {package}?",
            f"I'd like to know the public members of the {package}."
            " Can you provide that information?",
        ]
        package_members_answers = [
            f"{package} does not have any public member exported through '__all__'.",
            f"The {package} does not export any public members through '__all__'.",
            f"The {package} does not have any public members exported through '__all__'.",
        ]

        package_retrieval_chunks.append(package_members_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_members_context, package_members_questions, package_members_answers
            )
        )
    else:
        package_exports_count = len(package_exports)

        package_exports_count_context = (
            f"{package} has {package_exports_count} many public exports."
        )
        package_exports_count_questions = [
            f"How many objects does {package} export publicly?",
            f"What is the count of publicly exported objects in {package}?",
            f"Could you tell me the number of objects publicly exported by {package}?",
            f"Please provide the count of objects publicly exported by {package}.",
            f"Tell me the quantity of objects that {package} exports publicly.",
            f"Would you mind letting me know how many objects {package} publicly exports?",
        ]
        package_exports_count_answers = [
            f"{package} exports {package_exports_count} many objects using __all__.",
            f"Count of publicly exported objects in {package} is {package_exports_count}.",
            f"{package} exports {package_exports_count} objects using __all__.",
            f"Number of objects publicly exported by {package} is {package_exports_count}.",
            f"{package} exports {package_exports_count} objects using __all__.",
            f"{package} publicly exports {package_exports_count} objects.",
        ]

        package_retrieval_chunks.append(package_exports_count_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_exports_count_context,
                package_exports_count_questions,
                package_exports_count_answers,
            )
        )

        package_public_members = enumerate_array_elements(package_exports)

        package_members_context = (
            f"{package} exports following public members using __all__: {package_public_members}."
        )
        package_members_questions = [
            f"What are the public members of the {package}?",
            f"Can you list the public members of the {package}?",
            f"I need to know the public members of the {package}. Can you tell me?",
            f"Could you tell me what the {package} publicly exports?",
            f"I'm interested in the public members of the {package}. What are they?",
        ]
        package_members_answers = [
            f"{package} publicly exports the following members using '__all__':"
            f" {package_public_members}.",
            f"The {package} publicly exports the following members using '__all__':"
            f" {package_public_members}.",
            f"The {package} publicly exports these members using '__all__':"
            f" {package_public_members}.",
        ]

        package_retrieval_chunks.append(package_members_context)
        package_tuning_documents.extend(
            allocate_tuning_triplets(
                package_members_context, package_members_questions, package_members_answers
            )
        )

    package_dataset = Dataset(
        retrieval_chunks=package_retrieval_chunks, tuning_documents=package_tuning_documents
    )

    return package_dataset


@pydantic.validate_call(validate_return=True)
def generate_module_dataset(module_contents: ModuleDetails) -> Dataset:  # noqa: PLR0915
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
    module_tuning_documents: list[Document] = []

    module_package_context = (
        f"{module} is part of parent package '{module_contents.package_name}'."
    )
    module_package_questions = [
        f"Can you tell the the parent package of {module}?",
        f"What is the parent package of the {module}?",
        f"I'm trying to find the parent package of the {module}. Can you help?",
        f"Could you inform me about the parent package of the {module}?",
        f"I need to know the parent package of {module}. Can you provide that information?",
        f"Can you identify the parent package for the {module}?",
    ]
    module_package_answers = [
        f"'{module_contents.package_name}' is the parent package of {module}.",
        f"The parent package of {module} is '{module_contents.package_name}'.",
        f"Parent package of {module} is '{module_contents.package_name}'.",
        f"'{module_contents.package_name}' is the parent package of the {module}.",
        f"The parent package of the {module} is '{module_contents.package_name}'.",
        f"Parent package for {module} is '{module_contents.package_name}'.",
    ]

    module_retrieval_chunks.append(module_package_context)
    module_tuning_documents.extend(
        allocate_tuning_triplets(
            module_package_context, module_package_questions, module_package_answers
        )
    )

    module_full_name_context = f"Full name of {module} is '{module_full_name}'."
    module_full_name_questions = [
        f"Specify the full name of {module}?",
        f"What is the fully qualified name for the {module}?",
        f"Could you tell me the full name of the {module}?",
        f"I need the full name of the {module}. Can you provide it?",
        f"Can you specify the fully qualified name of the {module}?",
        f"I'm looking for the full name of the {module}. What is it?",
    ]
    module_full_name_answers = [
        f"'{module_full_name}' is fully qualified name for {module}.",
        f"The fully qualified name for the {module} is '{module_full_name}'.",
        f"The full name of the {module} is '{module_full_name}'.",
        f"Fully qualified name of the {module} is '{module_full_name}'.",
        f"Full name of the {module} you're looking for is '{module_full_name}'.",
    ]

    module_retrieval_chunks.append(module_full_name_context)
    module_tuning_documents.extend(
        allocate_tuning_triplets(
            module_full_name_context, module_full_name_questions, module_full_name_answers
        )
    )

    module_hierarchy = enumerate_array_elements(module_contents.module_hierarchy)

    module_hierarchy_context = f"Hierarchy of {module} is as follows: {module_hierarchy}."
    module_hierarchy_questions = [
        f"What is the hierarchy of {module}?",
        f"Can you explain the hierarchy of the {module}?",
        f"Could you describe the structure of the {module}?",
        f"I need to understand the hierarchy of the {module}. Can you help?",
        f"Please provide the hierarchy of the {module}.",
        f"What does the hierarchy of the {module} look like?",
    ]
    module_hierarchy_answers = [
        f"The hierarchy of {module} is as follows: {module_hierarchy}.",
        f"The hierarchy of the {module} is: {module_hierarchy}.",
        f"The structure of the {module} is: {module_hierarchy}.",
        f"The hierarchy of the {module} looks like this: {module_hierarchy}.",
    ]

    module_retrieval_chunks.append(module_hierarchy_context)
    module_tuning_documents.extend(
        allocate_tuning_triplets(
            module_hierarchy_context, module_hierarchy_questions, module_hierarchy_answers
        )
    )

    module_members_count = len(module_contents.module_members)

    module_members_count_context = f"{module} has {module_members_count} many members."
    module_members_count_questions = [
        f"How many members does {module} have?",
        f"What is the count of members in {module}?",
        f"Could you tell me the number of members in {module}?",
        f"Please provide the count of members for {module}.",
        f"Tell me the quantity of members present in {module}.",
        f"Would you mind letting me know how many members {module} contains?",
    ]
    module_members_count_answers = [
        f"{module} has {module_members_count} many members.",
        f"The count of members in {module} is {module_members_count}.",
        f"{module} has {module_members_count} members.",
        f"The number of members in {module} is {module_members_count}.",
        f"{module} contains {module_members_count} members.",
    ]

    module_retrieval_chunks.append(module_members_count_context)
    module_tuning_documents.extend(
        allocate_tuning_triplets(
            module_members_count_context,
            module_members_count_questions,
            module_members_count_answers,
        )
    )

    module_member_names = enumerate_array_elements(
        module_contents.module_members, attribute="member_name"
    )

    module_members_context = f"Members of {module} are as follows: {module_member_names}."
    module_members_questions = [
        f"List the members of {module}.",
        f"What are the members of the {module}?",
        f"Can you tell me the members of the {module}?",
        f"I need to know the members of the {module}.",
        f"Could you list the members of the {module}?",
        f"Please provide the members of the {module}.",
    ]
    module_members_answers = [
        f"Members of {module} are as follows: {module_member_names}.",
        f"The {module} has the following members: {module_member_names}.",
        f"The members of the {module} are: {module_member_names}.",
        f"Members of {module} you asked for are: {module_member_names}.",
        f"Members of the {module} are: {module_member_names}.",
        f"Members of {module} you requested are: {module_member_names}.",
    ]

    module_retrieval_chunks.append(module_members_context)
    module_tuning_documents.extend(
        allocate_tuning_triplets(
            module_members_context, module_members_questions, module_members_answers
        )
    )

    if not (module_summary := module_contents.module_summary):
        module_summary_context = (
            f"Unfortunately, {module} currently does not have any documentation."
        )
        module_summary_questions = [
            f"What is the {module} for?",
            f"Can you tell me the purpose of the {module}?",
            f"I'd like to know what the {module} is used for.",
            f"Could you explain the function of the {module}?",
            f"What does the {module} do?",
        ]
        module_summary_answers = [
            f"{module} does not have any documentation.",
            f"The {module} lacks any documentation.",
            f"There is no documentation for the {module}.",
            f"The {module} doesn't come with any documentation.",
            f"The {module} is without any documentation.",
        ]

        module_retrieval_chunks.append(module_summary_context)
        module_tuning_documents.extend(
            allocate_tuning_triplets(
                module_summary_context, module_summary_questions, module_summary_answers
            )
        )
    else:
        module_summary_context = (
            f"The following is the documentation of {module}: {module_summary}."
        )
        module_summary_questions = [
            f"What is the '{module_name}' module for?",
            f"Can you tell me the purpose of the '{module_name}' module?",
            f"I'm curious about the '{module_name}' module. What does it do?",
            f"Could you explain the functionality of the '{module_name}' module?",
            f"I'd like to know more about the '{module_name}' module. What's its role?",
            f"What's the use of the '{module_name}' module?",
        ]
        module_summary_answers = [
            f"{module} documents itself as follows: '{module_summary}'.",
            f"Purpose of {module} is documented as: '{module_summary}'.",
            f"The {module} is described as: '{module_summary}'.",
            f"The functionality of the {module} is described as: '{module_summary}'.",
            f"The role of the {module} is: '{module_summary}'.",
            f"Use of the {module} is documented as: '{module_summary}'.",
        ]

        module_retrieval_chunks.append(module_summary_context)
        module_tuning_documents.extend(
            allocate_tuning_triplets(
                module_summary_context, module_summary_questions, module_summary_answers
            )
        )

    if not (module_exports := module_contents.module_all_exports):
        module_exports_context = (
            f"{module} does not export anything publicly using __all__ variable."
        )
        module_exports_questions = [
            f"Tell me the public members of the {module}.",
            f"What are the public members of the {module}?",
            f"Could you list the public members of the {module}?",
            f"I need to know the public members of the {module}.",
            f"Can you show me the public members of the {module}?",
            f"I'm interested in the public members of the {module}. What are they?",
        ]
        module_exports_answers = [
            f"{module} lacks any public member exported through '__all__'.",
            f"There are no public members exported through '__all__' in the {module}.",
            f"{module} does not export any public members through '__all__'.",
            f"The {module} does not have any public members exported through '__all__'.",
            f"The {module} does not contain any public members exported through '__all__'.",
            f"{module} does not export any public members through '__all__'.",
        ]

        module_retrieval_chunks.append(module_exports_context)
        module_tuning_documents.extend(
            allocate_tuning_triplets(
                module_exports_context, module_exports_questions, module_exports_answers
            )
        )
    else:
        module_exports_count = len(module_exports)

        module_exports_count_context = f"{module} has {module_exports_count} many public exports."
        module_exports_count_questions = [
            f"How many objects does {module} export publicly?",
            f"What is the count of publicly exported objects in {module}?",
            f"Could you tell me the number of objects publicly exported by {module}?",
            f"Please provide the count of objects publicly exported by {module}.",
            f"Tell me the quantity of objects that {module} exports publicly.",
            f"Would you mind letting me know how many objects {module} publicly exports?",
        ]
        module_exports_count_answers = [
            f"{module} exports {module_exports_count} many objects using __all__.",
            f"The count of publicly exported objects in {module} is {module_exports_count}.",
            f"{module} exports {module_exports_count} objects using __all__.",
            f"The number of objects publicly exported by {module} is {module_exports_count}.",
            f"{module} exports {module_exports_count} objects using __all__.",
            f"{module} publicly exports {module_exports_count} objects.",
        ]

        module_retrieval_chunks.append(module_exports_count_context)
        module_tuning_documents.extend(
            allocate_tuning_triplets(
                module_exports_count_context,
                module_exports_count_questions,
                module_exports_count_answers,
            )
        )

        module_public_exports = enumerate_array_elements(module_exports)

        module_exports_context = (
            f"{module} exports following members using __all__: {module_public_exports}."
        )
        module_exports_questions = [
            f"Tell me the public members of the {module}.",
            f"What are the public members of the {module}?",
            f"Could you list the public members of the {module}?",
            f"I need to know the public members of the {module}.",
            f"Can you show me the public members of the {module}?",
        ]
        module_exports_answers = [
            f"{module} publicly exports the following members using '__all__':"
            f" {module_public_exports}.",
            f"The {module} publicly exports the following members using '__all__':"
            f" {module_public_exports}.",
            f"The {module} publicly exports these members using '__all__':"
            f" {module_public_exports}.",
        ]

        module_retrieval_chunks.append(module_exports_context)
        module_tuning_documents.extend(
            allocate_tuning_triplets(
                module_exports_context, module_exports_questions, module_exports_answers
            )
        )

    module_dataset = Dataset(
        retrieval_chunks=module_retrieval_chunks, tuning_documents=module_tuning_documents
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
    enum_member_tuning_documents: list[Document] = []

    enum_member_count = len(member_type_details.enum_members)

    enum_member_count_context = f"{enum_member} has {enum_member_count} many members."
    enum_member_count_questions = [
        f"How many members are there in {enum_member}?",
        f"What is the count of members in {enum_member}?",
        f"Can you tell me the number of members in {enum_member}?",
        f"Could you provide the total number of members in {enum_member}?",
        f"I need to know the quantity of members in {enum_member}.",
        f"Please inform me about the number of members in {enum_member}.",
    ]
    enum_member_count_answers = [
        f"{enum_member} has {enum_member_count} members.",
        f"The count of members in {enum_member} is {enum_member_count}.",
        f"The number of members in {enum_member} is {enum_member_count}.",
        f"The total number of members in {enum_member} is {enum_member_count}.",
        f"The quantity of members in {enum_member} is {enum_member_count}.",
        f"The number of members in {enum_member} is {enum_member_count}.",
    ]

    enum_member_retrieval_chunks.append(enum_member_count_context)
    enum_member_tuning_documents.extend(
        allocate_tuning_triplets(
            enum_member_count_context, enum_member_count_questions, enum_member_count_answers
        )
    )

    enum_members = enumerate_array_elements(
        member_type_details.enum_members, attribute="enum_member"
    )

    enum_members_context = f"Members of {enum_member} are as follows: {enum_members}."
    enum_members_questions = [
        f"What are the different members of {enum_member}?",
        f"Can you list the different members of {enum_member}?",
        f"Could you tell me the different members of {enum_member}?",
        f"I need to know the different members of {enum_member}.",
        f"What does {enum_member} consist of?",
    ]
    enum_members_answers = [
        f"Different members of {enum_member} are as follows: {enum_members}.",
        f"The different members of {enum_member} include: {enum_members}.",
        f"The different members of {enum_member} are: {enum_members}.",
        f"{enum_member} consists of the following members: {enum_members}.",
    ]

    enum_member_retrieval_chunks.append(enum_members_context)
    enum_member_tuning_documents.extend(
        allocate_tuning_triplets(
            enum_members_context, enum_members_questions, enum_members_answers
        )
    )

    enum_member_names = enumerate_array_elements(
        member_type_details.enum_members, attribute="enum_member_name"
    )

    enum_member_names_context = (
        f"Names of different members of {enum_member} are as follows: {enum_member_names}."
    )
    enum_member_names_questions = [
        f"List just the names of different members of {enum_member}.",
        f"Can you provide the names of different members of {enum_member}?",
        f"What are the names of different members of {enum_member}?",
        f"I need the names of different members of {enum_member}.",
        f"Could you list the names of different members of {enum_member}?",
        f"Show me the names of different members of {enum_member}.",
    ]
    enum_member_names_answers = [
        f"Different members of {enum_member} have the following names: {enum_member_names}.",
        f"Different members of {enum_member} are named as follows: {enum_member_names}.",
        f"The names of different members of {enum_member} are: {enum_member_names}.",
        f"Different members of {enum_member} have these names: {enum_member_names}.",
    ]

    enum_member_retrieval_chunks.append(enum_member_names_context)
    enum_member_tuning_documents.extend(
        allocate_tuning_triplets(
            enum_member_names_context, enum_member_names_questions, enum_member_names_answers
        )
    )

    enum_member_values = enumerate_array_elements(
        member_type_details.enum_members, attribute="enum_member_value"
    )

    enum_member_values_context = (
        f"Values of different members of {enum_member} are as follows: {enum_member_values}."
    )
    enum_member_values_questions = [
        f"Only show the different values supported by {enum_member}.",
        f"What are the different values that {enum_member} supports?",
        f"Can you list the values supported by {enum_member}?",
        f"I need to know the values supported by {enum_member}.",
        f"Could you tell me the values that {enum_member} supports?",
        f"Please provide the values supported by {enum_member}.",
    ]
    enum_member_values_answers = [
        f"{enum_member} supports the following values: {enum_member_values}.",
        f"The different values that {enum_member} supports are: {enum_member_values}.",
        f"{enum_member} supports these values: {enum_member_values}.",
        f"The values that {enum_member} supports are: {enum_member_values}.",
        f"The values supported by {enum_member} are: {enum_member_values}.",
    ]

    enum_member_retrieval_chunks.append(enum_member_values_context)
    enum_member_tuning_documents.extend(
        allocate_tuning_triplets(
            enum_member_values_context, enum_member_values_questions, enum_member_values_answers
        )
    )

    enum_member_dataset = Dataset(
        retrieval_chunks=enum_member_retrieval_chunks,
        tuning_documents=enum_member_tuning_documents,
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
    class_member_tuning_documents: list[Document] = []

    if not (class_parameters := member_type_details.class_parameters):
        class_parameters_context = f"{class_member} requires no arguments for instantiation."
        class_parameters_questions = [
            f"What are the different parameters of {class_member}?",
            f"Can you tell me the parameters required for {class_member}?",
            f"What arguments do I need to instantiate {class_member}?",
            f"Do I need any parameters to use {class_member}?",
            f"What should I pass as arguments when creating an instance of {class_member}?",
            f"Are there any parameters needed for the instantiation of {class_member}?",
        ]
        class_parameters_answers = [
            f"{class_member} needs no arguments for instantiation.",
            f"No parameters are required for instantiating {class_member}.",
            f"Arguments are not needed to instantiate {class_member}.",
            f"{class_member} can be used without any parameters.",
            f"There's no need to pass any arguments when creating an instance of {class_member}.",
            f"The instantiation of {class_member} doesn't require any parameters.",
        ]

        class_member_retrieval_chunks.append(class_parameters_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_parameters_context, class_parameters_questions, class_parameters_answers
            )
        )
    else:
        class_parameter_names = enumerate_array_elements(
            class_parameters, attribute="parameter_details"
        )

        class_parameters_context = (
            f"{class_member} requires the following arguments for initialisation:"
            f" {class_parameter_names}"
        )
        class_parameters_questions = [
            f"What are the different parameters of {class_member}?",
            f"Can you list the parameters for {class_member}?",
            f"I need to know the parameters of {class_member}.",
            f"Tell me the parameters that {class_member} supports.",
            f"What arguments does {class_member} take for initialisation?",
        ]
        class_parameters_answers = [
            f"{class_member} supports these arguments to initiate"
            f" a new instance: {class_parameter_names}.",
            f"{class_member} can be initiated with these arguments: {class_parameter_names}.",
            f"The parameters to initiate a new instance of {class_member} are:"
            f" {class_parameter_names}.",
            f"To initialise {class_member}, you can use these arguments: {class_parameter_names}.",
        ]

        class_member_retrieval_chunks.append(class_parameters_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_parameters_context, class_parameters_questions, class_parameters_answers
            )
        )

    for class_parameter in class_parameters:
        parameter_name = class_parameter.parameter_name
        parameter = f"'{parameter_name}' argument in {class_member}"

        if (parameter_default := class_parameter.parameter_default) is EMPTY_PARAMETER:
            class_parameter_defaults_context = f"{parameter} does not have a default value."
            class_parameter_defaults_questions = [
                f"Tell default value of {parameter}.",
                f"What is the default value of {parameter}?",
                f"Could you inform me about default value of {parameter}?",
                f"I need to know the default value of {parameter}. Can you help?",
                f"Can you tell me if {parameter} has default value?",
                f"I'm curious about default value of {parameter}.",
            ]
            class_parameter_defaults_answers = [
                f"{parameter} does not have a default value.",
                f"The {parameter} does not have a default value.",
            ]

            class_member_retrieval_chunks.append(class_parameter_defaults_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_parameter_defaults_context,
                    class_parameter_defaults_questions,
                    class_parameter_defaults_answers,
                )
            )
        else:
            class_parameter_defaults_context = (
                f"{parameter_default} is the default value of {parameter}."
            )
            class_parameter_defaults_questions = [
                f"Tell default value of {parameter}.",
                f"What is the default value of {parameter}?",
                f"Could you inform me about default value of {parameter}?",
                f"I need to know the default value of {parameter}.",
                f"Can you provide default value of {parameter}?",
                f"Please, disclose default value of {parameter}.",
            ]
            class_parameter_defaults_answers = [
                f"{parameter} takes {parameter_default} by default.",
                f"Default value of {parameter} is {parameter_default}.",
                f"The default value of {parameter} is {parameter_default}.",
            ]

            class_member_retrieval_chunks.append(class_parameter_defaults_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_parameter_defaults_context,
                    class_parameter_defaults_questions,
                    class_parameter_defaults_answers,
                )
            )

        if (parameter_annotation := class_parameter.parameter_annotation) is EMPTY_PARAMETER:
            class_parameter_types_context = f"Type hint for {parameter} is unavailable."
            class_parameter_types_questions = [
                f"Name type hint for {parameter}.",
                f"What is the type hint for {parameter}?",
                f"Can you tell me the type hint for {parameter}?",
                f"I'm looking for the type hint for {parameter}. Can you help?",
                f"Could you provide the type hint for {parameter}?",
                f"I need to know the type hint for {parameter}.",
            ]
            class_parameter_types_answers = [
                f"{parameter} does not have a type annotation.",
                f"There is no type annotation for the {parameter}.",
                f"The {parameter} is not annotated with a type.",
                f"The {parameter} does not have a type annotation.",
                f"{parameter} does not have type annotation.",
                f"The {parameter} does not come with a type annotation.",
            ]

            class_member_retrieval_chunks.append(class_parameter_types_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_parameter_types_context,
                    class_parameter_types_questions,
                    class_parameter_types_answers,
                )
            )
        else:
            class_parameter_types_context = (
                f"{parameter} is annotated as '{parameter_annotation}' type."
            )
            class_parameter_types_questions = [
                f"Name type hint for {parameter}.",
                f"What is the type hint for {parameter}?",
                f"Could you tell me the type hint for {parameter}?",
                f"I need to know the type hint for {parameter}.",
                f"Identify the type hint for {parameter}.",
                f"Can you specify the type hint for {parameter}?",
            ]
            class_parameter_types_answers = [
                f"{parameter} has '{parameter_annotation}' as type hint.",
                f"The type hint for {parameter} is '{parameter_annotation}'.",
            ]

            class_member_retrieval_chunks.append(class_parameter_types_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_parameter_types_context,
                    class_parameter_types_questions,
                    class_parameter_types_answers,
                )
            )

        if not (parameter_summary := class_parameter.parameter_summary):
            class_parameter_summary_context = (
                f"{parameter} lacks any documentation in the docstring."
            )
            class_parameter_summary_questions = [
                f"What does {parameter} do?",
                f"Can you explain the role of {parameter}?",
                f"I'm trying to understand what {parameter} does. Can you help?",
                f"What is the function of {parameter}?",
                f"Could you tell me what '{parameter_name}' does in {class_member}?",
                f"I'm curious about the purpose of {parameter}. Can you enlighten me?",
            ]
            class_parameter_summary_answers = [
                f"Docstring of {class_member} does not describe '{parameter_name}'.",
                f"The docstring of {class_member} does not provide any information about"
                f" '{parameter_name}'.",
                f"The docstring of {class_member} does not mention anything"
                f" about '{parameter_name}'.",
                f"There is no description of '{parameter_name}' in the docstring of"
                f" {class_member}.",
                f"The docstring of {class_member} does not contain any details about"
                f" '{parameter_name}'.",
                f"The docstring of {class_member} does not discuss '{parameter_name}'.",
            ]

            class_member_retrieval_chunks.append(class_parameter_summary_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_parameter_summary_context,
                    class_parameter_summary_questions,
                    class_parameter_summary_answers,
                )
            )
        else:
            class_parameter_summary_context = (
                f"As per docstring, role of {parameter} is: '{parameter_summary}'."
            )
            class_parameter_summary_questions = [
                f"What does {parameter} do?",
                f"Can you explain the role of {parameter}?",
                f"I'm curious about {parameter}. What does it do?",
                f"Could you tell me what {parameter} does?",
                f"What's the function of {parameter}?",
                f"I'd like to know the purpose of {parameter}.",
            ]
            class_parameter_summary_answers = [
                f"{class_member} documents role of '{parameter_name}' as follows:"
                f" '{parameter_summary}'.",
                f"{class_member} describes '{parameter_name}' as follows: '{parameter_summary}'.",
                f"In {class_member}, '{parameter_name}' is documented as follows:"
                f" '{parameter_summary}'.",
                f"{parameter} is described as follows: '{parameter_summary}'.",
                f"{class_member} describes the function of '{parameter_name}' as follows:"
                f" '{parameter_summary}'.",
                f"In {class_member}, the purpose of '{parameter_name}' is defined as follows:"
                f" '{parameter_summary}'.",
            ]

            class_member_retrieval_chunks.append(class_parameter_summary_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_parameter_summary_context,
                    class_parameter_summary_questions,
                    class_parameter_summary_answers,
                )
            )

    if not (class_methods := member_type_details.class_methods):
        class_method_names_context = (
            f"{class_member} has no public (without _ as the prefix) methods."
        )
        class_method_names_questions = [
            f"List names of the public methods of {class_member}.",
            f"Can you provide the names of the public methods for {class_member}?",
            f"What are the public methods of {class_member}?",
            f"I need to know the public methods of {class_member}. Can you list them?",
            f"Could you list the public methods of {class_member}?",
            f"Show me the public methods of {class_member}.",
        ]
        class_method_names_answers = [
            f"{class_member} does not have any public methods (not starting with '_').",
            f"{class_member} does not have any public methods.",
            f"There are no public methods (not starting with '_') in {class_member}.",
            f"{class_member} does not have any public methods.",
            f"{class_member} does not contain any public methods (not starting with '_').",
            f"It appears that {class_member} does not have any public methods.",
        ]

        class_member_retrieval_chunks.append(class_method_names_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_method_names_context,
                class_method_names_questions,
                class_method_names_answers,
            )
        )
    else:
        class_methods_count = len(class_methods)

        class_methods_count_context = (
            f"{class_member} has {class_methods_count} many public methods."
        )
        class_methods_count_questions = [
            f"How many public methods does {class_member} have?",
            f"What is the count of public methods in {class_member}?",
            f"Could you tell me the number of public methods in {class_member}?",
            f"Please provide the count of public methods for {class_member}.",
            f"Tell me the quantity of public methods present in {class_member}.",
            f"Would you mind letting me know how many public methods {class_member} contains?",
        ]
        class_methods_count_answers = [
            f"{class_member} has {class_methods_count} many public methods.",
            f"The count of public methods in {class_member} is {class_methods_count}.",
            f"{class_member} has {class_methods_count} public methods.",
            f"The number of public methods in {class_member} is {class_methods_count}.",
            f"{class_member} has {class_methods_count} public methods.",
            f"{class_member} contains {class_methods_count} public methods.",
        ]

        class_member_retrieval_chunks.append(class_methods_count_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_methods_count_context,
                class_methods_count_questions,
                class_methods_count_answers,
            )
        )

        class_public_methods = enumerate_array_elements(class_methods, attribute="method_name")

        class_method_names_context = (
            f"{class_member} has the following public methods: {class_public_methods}"
        )
        class_method_names_questions = [
            f"List names of the public methods of {class_member}.",
            f"Can you provide the names of the public methods for {class_member}?",
            f"What are the public methods of {class_member}?",
            f"I need to know the public methods of {class_member}.",
            f"Could you list the public methods of {class_member}?",
            f"Please show me the public methods of {class_member}.",
        ]
        class_method_names_answers = [
            f"Here are the public methods of {class_member}: {class_public_methods}.",
            f"The public methods of {class_member} that do not start with '_' are:"
            f" {class_public_methods}.",
            f"The public methods of {class_member} (excluding those starting with '_') are:"
            f" {class_public_methods}.",
            f"The public methods of {class_member} (those not starting with '_') are:"
            f" {class_public_methods}.",
            f"The public methods of {class_member} (not beginning with '_') are:"
            f" {class_public_methods}.",
            f"The public methods of {class_member} (excluding those with a prefix '_') are:"
            f" {class_public_methods}.",
        ]

        class_member_retrieval_chunks.append(class_method_names_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_method_names_context,
                class_method_names_questions,
                class_method_names_answers,
            )
        )

    for class_method in class_methods:
        method_name = class_method.method_name
        method = f"'{method_name}' method of {class_member}"

        if not (method_parameters := class_method.method_parameters):
            class_method_parameters_context = f"{method} takes no arguments."
            class_method_parameters_questions = [
                f"What arguments do {method} accept?",
                f"Can you tell me the parameters that {method} requires?",
                f"What are the inputs for the {method} in {class_member}?",
                f"Does the {method} need any arguments?",
                f"What parameters should I pass to {method}?",
                f"What are required arguments for {method}?",
            ]
            class_method_parameters_answers = [
                f"{method} does not take any parameters.",
                f"The {method} does not require any parameters.",
                f"There are no inputs for the {method} in {class_member}.",
                f"{method} does not need any arguments.",
                f"No parameters need to be passed to the {method}.",
                f"{method} does not require any arguments.",
            ]

            class_member_retrieval_chunks.append(class_method_parameters_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_method_parameters_context,
                    class_method_parameters_questions,
                    class_method_parameters_answers,
                )
            )
        else:
            class_method_parameters = enumerate_array_elements(method_parameters)

            class_method_parameters_context = (
                f"{method} accepts following parameters: {class_method_parameters}"
            )
            class_method_parameters_questions = [
                f"What arguments do {method} accept?",
                f"Can you tell me the parameters that {method} requires?",
                f"I need to know arguments for {method}.",
                f"What are the parameters for '{method}'?",
                f"Could you list the arguments that the {method} takes?",
            ]
            class_method_parameters_answers = [
                f"{method} takes the following parameters: {class_method_parameters}.",
                f"{method} requires these parameters: {class_method_parameters}.",
                f"The {method} has these arguments: {class_method_parameters}.",
                f"The parameters for {method} are: {class_method_parameters}.",
                f"The {method} takes these arguments: {class_method_parameters}.",
            ]

            class_member_retrieval_chunks.append(class_method_parameters_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_method_parameters_context,
                    class_method_parameters_questions,
                    class_method_parameters_answers,
                )
            )

        if not (method_summary := class_method.method_summary):
            class_method_summary_context = f"Unfortunately, {method} is not documented."
            class_method_summary_questions = [
                f"What does {method} do?",
                f"Can you explain functionality of {method}?",
                f"I'm trying to understand what {method} does. Can you help?",
                f"Could you describe the role of {method}?",
                f"I'm not sure what {method} does. Can you clarify?",
                f"What's the purpose of {method}?",
            ]
            class_method_summary_answers = [
                f"Docstring of {method} is missing.",
                f"The docstring for {method} is not available.",
                f"The docstring for {method} is not provided.",
                f"There is no docstring available for {method}.",
                f"The {method} lacks a docstring.",
                f"The {method} doesn't have a docstring.",
            ]

            class_member_retrieval_chunks.append(class_method_summary_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_method_summary_context,
                    class_method_summary_questions,
                    class_method_summary_answers,
                )
            )
        else:
            class_method_summary_context = (
                f"Based on docstring, {method} has the purpose of '{method_summary}'."
            )
            class_method_summary_questions = [
                f"What does {method} do?",
                f"Can you explain the function of {method}?",
                f"I'm curious about the {method}. What's its purpose?",
                f"Could you tell me what the {method} does?",
                f"I'd like to understand role of {method}.",
                f"What's the functionality of the {method}?",
            ]
            class_method_summary_answers = [
                f"Based on method docstring, its role is to '{method_summary}'.",
                f"According to method docstring, it is designed to '{method_summary}'.",
                f"If we look at the docstring of {method}, we can see that it's meant to"
                f" '{method_summary}'.",
                f"The docstring of {method} indicates that its function is to '{method_summary}'.",
                f"Method docstring reveals that its job is to '{method_summary}'.",
                f"As per the method docstring, it's designed to '{method_summary}'.",
            ]

            class_member_retrieval_chunks.append(class_method_summary_context)
            class_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    class_method_summary_context,
                    class_method_summary_questions,
                    class_method_summary_answers,
                )
            )

    if not (class_attributes := member_type_details.class_attributes):
        class_attribute_names_context = f"{class_member} has no public attributes."
        class_attribute_names_questions = [
            f"Are there any public attributes of {class_member}?",
            f"Does {class_member} have any public attributes?",
            f"Can you tell me if {class_member} has any public attributes?",
            f"I'm looking for public attributes of {class_member}. Are there any?",
            f"Is it possible to find any public attributes in {class_member}?",
        ]
        class_attribute_names_answers = [
            f"{class_member} has no public attributes (not starting with '_').",
            f"{class_member} does not have any public attributes.",
            f"{class_member} does not have any public attributes (not starting with '_').",
            f"There are no public attributes (not starting with '_') for {class_member}.",
            f"It's not possible to find any public attributes in {class_member}.",
        ]

        class_member_retrieval_chunks.append(class_attribute_names_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_attribute_names_context,
                class_attribute_names_questions,
                class_attribute_names_answers,
            )
        )
    else:
        class_attributes_count = len(class_attributes)

        class_attributes_count_context = (
            f"{class_member} has {class_attributes_count} many public attributes."
        )
        class_attributes_count_questions = [
            f"How many public attributes does {class_member} have?",
            f"What is the count of public attributes in {class_member}?",
            f"Could you tell me the number of public attributes in {class_member}?",
            f"Please provide the count of public attributes for {class_member}.",
            f"Tell me the quantity of public attributes present in {class_member}.",
            f"Would you mind letting me know how many public attributes {class_member} contains?",
        ]
        class_attributes_count_answers = [
            f"{class_member} has {class_attributes_count} many public attributes.",
            f"The count of public attributes in {class_member} is {class_attributes_count}.",
            f"{class_member} has {class_attributes_count} public attributes.",
            f"Number of public attributes in {class_member} is {class_attributes_count}.",
            f"{class_member} has {class_attributes_count} public attributes.",
            f"{class_member} contains {class_attributes_count} public attributes.",
        ]

        class_member_retrieval_chunks.append(class_attributes_count_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_attributes_count_context,
                class_attributes_count_questions,
                class_attributes_count_answers,
            )
        )

        class_public_attributes = enumerate_array_elements(
            class_attributes, attribute="attribute_name"
        )

        class_attribute_names_context = (
            f"{class_member} has following public attributes: {class_public_attributes}"
        )
        class_attribute_names_questions = [
            f"Are there any public attributes of {class_member}?",
            f"Can you list the public attributes of {class_member}?",
            f"What are the public attributes of {class_member}?",
            f"I need to know the public attributes of {class_member}.",
            f"Could you tell me the public attributes of {class_member}?",
        ]
        class_attribute_names_answers = [
            f"These are the public attributes of {class_member}: {class_public_attributes}.",
            f"{class_member} has the following public attributes (not starting with '_'):"
            f" {class_public_attributes}.",
            f"The public attributes of {class_member} (those not starting with '_') are:"
            f" {class_public_attributes}.",
            f"The public attributes of {class_member} are: {class_public_attributes}.",
            f"Public attributes of {class_member} (not starting with '_') are:"
            f" {class_public_attributes}.",
        ]

        class_member_retrieval_chunks.append(class_attribute_names_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_attribute_names_context,
                class_attribute_names_questions,
                class_attribute_names_answers,
            )
        )

    if not (class_summary := member_type_details.class_summary):
        class_summary_context = f"Unfortunately, {class_member} does not document its objective."
        class_summary_questions = [
            f"What does {class_member} do in short?",
            f"Can you briefly explain the function of {class_member}?",
            f"Could you tell me what {class_member} is used for?",
            f"I'm not sure what {class_member} does. Can you clarify?",
            f"What's the purpose of {class_member}?",
        ]
        class_summary_answers = [
            f"Docstring of {class_member} lacks a summary of its objective.",
            f"Docstring of {class_member} doesn't provide a concise summary of its purpose.",
            f"The docstring of {class_member} doesn't contain"
            " a brief description of its function.",
            f"The docstring of {class_member} doesn't succinctly explain its role.",
            f"Docstring of {class_member} doesn't have any explanation of its objective.",
        ]

        class_member_retrieval_chunks.append(class_summary_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_summary_context, class_summary_questions, class_summary_answers
            )
        )
    else:
        class_summary_context = (
            f"{class_member} documents its purpose as follows: '{class_summary}'."
        )
        class_summary_questions = [
            f"What does {class_member} do in short?",
            f"Can you briefly explain the function of {class_member}?",
            f"I'm curious about {class_member}, what's its purpose?",
            f"Could you give me a quick rundown on what {class_member} does?",
            f"What's the role of {class_member} in a nutshell?",
            f"Can you summarise the function of {class_member}?",
        ]
        class_summary_answers = [
            f"Based on documentation, objective of {class_member} is to: '{class_summary}'.",
            f"According to the documentation, {class_member} is designed to: '{class_summary}'.",
            f"As per the documentation, {class_member} aims to: '{class_summary}'.",
            f"The documentation states that the role of {class_member} is to: '{class_summary}'.",
            f"The documentation indicates that the purpose of {class_member} is to:"
            f" '{class_summary}'.",
            f"The documentation outlines that {class_member} is intended to: '{class_summary}'.",
        ]

        class_member_retrieval_chunks.append(class_summary_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_summary_context, class_summary_questions, class_summary_answers
            )
        )

    if not (class_notes := member_type_details.class_notes):
        class_notes_context = (
            f"Docstring of {class_member} has contains no specific implementation details."
        )
        class_notes_questions = [
            f"Mention any specific details for {class_member} to be aware of.",
            f"What are the specific details to be aware of for {class_member}?",
            f"Could you tell me any specifics for {class_member} that I should be aware of?",
            f"Are there any specific details for {class_member} that I need to know?",
            f"I need to know the specific details for {class_member}. Can you provide them?",
            f"Can you specify any details for {class_member} that I should be aware of?",
        ]
        class_notes_answers = [
            f"Docstring of {class_member} does not note on specific details.",
            f"There are no specific details noted in the docstring of {class_member}.",
            f"The docstring of {class_member} doesn't highlight any details.",
            f"No specific details are mentioned in the docstring of {class_member}.",
            f"The docstring of {class_member} does not contain any details.",
            f"The docstring of {class_member} does not specify any details to be aware of.",
        ]

        class_member_retrieval_chunks.append(class_notes_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_notes_context, class_notes_questions, class_notes_answers
            )
        )
    else:
        class_notes_context = (
            f"In docstring, {class_member} specifies the following: '{class_notes}'."
        )
        class_notes_questions = [
            f"Mention any specific details for {class_member} to be aware of.",
            f"What are specifics that I should be aware of before using {class_member}?",
            f"Could you specify the details for {class_member} to take note of?",
            f"Can you list the details for {class_member} to keep in mind?",
            f"What should users of {class_member} be mindful of?",
            f"What details does the user of {class_member} need to know?",
        ]
        class_notes_answers = [
            f"The {class_member} docstring highlights the following: '{class_notes}'.",
            f"The details you should know to use {class_member} are highlighted in docstring:"
            f" '{class_notes}'.",
            f"The docstring for {class_member} specifies the following details: '{class_notes}'.",
            f"The docstring for {class_member} lists the following details: '{class_notes}'.",
            f"The docstring for {class_member} mentions the following points to be mindful of:"
            f" '{class_notes}'.",
            f"User of {class_member} needs to know the following details: '{class_notes}'.",
        ]

        class_member_retrieval_chunks.append(class_notes_context)
        class_member_tuning_documents.extend(
            allocate_tuning_triplets(
                class_notes_context, class_notes_questions, class_notes_answers
            )
        )

    class_member_dataset = Dataset(
        retrieval_chunks=class_member_retrieval_chunks[:2],
        tuning_documents=class_member_tuning_documents,
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
    function_member_tuning_documents: list[Document] = []

    if not (function_parameters := member_type_details.function_parameters):
        function_parameters_context = f"{function_member} takes no parameters."
        function_parameters_questions = [
            f"List various parameters of {function_member}.",
            f"What are the parameters of {function_member}?",
            f"Could you tell me the parameters that {function_member} takes?",
            f"I need to know the parameters for {function_member}.",
            f"Can you list the parameters for {function_member}?",
            f"Please provide the parameters of {function_member}.",
        ]
        function_parameters_answers = [
            f"{function_member} does not take any parameters.",
            f"{function_member} has no parameters.",
            f"{function_member} doesn't require any parameters.",
            f"There are no parameters for {function_member}.",
            f"Actually, {function_member} doesn't have any parameters.",
            f"Sorry, but {function_member} does not have any parameters.",
        ]

        function_member_retrieval_chunks.append(function_parameters_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_parameters_context,
                function_parameters_questions,
                function_parameters_answers,
            )
        )
    else:
        function_parameter_names = enumerate_array_elements(
            function_parameters, attribute="parameter_details"
        )

        function_parameters_context = (
            f"{function_member} takes the following parameters: {function_parameter_names}"
        )
        function_parameters_questions = [
            f"List various parameters of {function_member}.",
            f"What are the different parameters of {function_member}?",
            f"Could you tell me the parameters of {function_member}?",
            f"I need to know the parameters of {function_member}.",
            f"Can you list the parameters for {function_member}?",
            f"Please provide the parameters of {function_member}.",
        ]
        function_parameters_answers = [
            f"Different parameters of {function_member} are as follows:"
            f" {function_parameter_names}.",
            f"{function_member} has the following parameters: {function_parameter_names}.",
            f"The parameters of {function_member} are: {function_parameter_names}.",
            f"Yes, the parameters for {function_member} are: {function_parameter_names}.",
            f"Parameters of {function_member} are as follows: {function_parameter_names}.",
        ]

        function_member_retrieval_chunks.append(function_parameters_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_parameters_context,
                function_parameters_questions,
                function_parameters_answers,
            )
        )

    for function_parameter in function_parameters:
        parameter_name = function_parameter.parameter_name
        parameter = f"'{parameter_name}' argument in {function_member}"

        if (parameter_default := function_parameter.parameter_default) is EMPTY_PARAMETER:
            function_parameter_defaults_context = f"{parameter} has no default value."
            function_parameter_defaults_questions = [
                f"Default value of {parameter}?",
                f"What is the default value for {parameter}?",
                f"Could you tell me default value of {parameter}?",
                f"I'm curious about default value of {parameter}.",
                f"I'd like to know the default value of {parameter}.",
                f"Can you inform me about the default value of {parameter}?",
            ]
            function_parameter_defaults_answers = [
                f"{parameter} does not have a default value."
                f"The {parameter} does not come with a default value.",
                f"The {parameter} does not possess a default value.",
                f"In response to your curiosity, {parameter} is not assigned a default value.",
                f"To answer your query, {parameter} does not hold a default value.",
                f"{parameter} does not contain a default value.",
            ]

            function_member_retrieval_chunks.append(function_parameter_defaults_context)
            function_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    function_parameter_defaults_context,
                    function_parameter_defaults_questions,
                    function_parameter_defaults_answers,
                )
            )
        else:
            function_parameter_defaults_context = (
                f"{parameter} has the default value of {parameter_default}."
            )
            function_parameter_defaults_questions = [
                f"Default value of {parameter}?",
                f"What is the default value for {parameter}?",
                f"Could you tell me default value of {parameter}?",
                f"I would like to know the default value of {parameter}.",
                f"Can you inform me about the default value of {parameter}?",
                f"I'm interested in default value of {parameter}.",
            ]
            function_parameter_defaults_answers = [
                f"{parameter} has default value of {parameter_default}.",
                f"The default value for {parameter} is {parameter_default}.",
                f"The default value of {parameter} is {parameter_default}.",
                f"The {parameter} has a default value of {parameter_default}.",
                f"The {parameter} defaults to {parameter_default}.",
                f"The default value of the {parameter} is {parameter_default}.",
            ]

            function_member_retrieval_chunks.append(function_parameter_defaults_context)
            function_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    function_parameter_defaults_context,
                    function_parameter_defaults_questions,
                    function_parameter_defaults_answers,
                )
            )

        if (parameter_annotation := function_parameter.parameter_annotation) is EMPTY_PARAMETER:
            function_parameter_types_context = (
                f"Unfortunately, type hint for {parameter} is missing."
            )
            function_parameter_types_questions = [
                f"What is type annotation of {parameter}?",
                f"Can you tell me type annotation of {parameter}?",
                f"I'm curious about the type annotation of {parameter}."
                " Can you provide some information?",
                f"Do you have any information on the type annotation of {parameter}?",
                f"Could you inform me about the type annotation of {parameter}?",
                f"I'd like to know the type annotation of {parameter}.",
            ]
            function_parameter_types_answers = [
                f"{parameter} does not have a type annotation.",
                f"The {parameter} does not have a type annotation.",
                f"{parameter} does not have a type annotation.",
                f"The {parameter} you're asking about does not have a type annotation.",
            ]

            function_member_retrieval_chunks.append(function_parameter_types_context)
            function_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    function_parameter_types_context,
                    function_parameter_types_questions,
                    function_parameter_types_answers,
                )
            )
        else:
            function_parameter_types_context = (
                f"{parameter} has '{parameter_annotation}' as type annotation."
            )
            function_parameter_types_questions = [
                f"What is type annotation of {parameter}?",
                f"Can you tell me type annotation of {parameter}?",
                f"I'm curious about the type annotation of {parameter}. What is it?",
                f"Do you know type annotation of {parameter}?",
                f"Could you inform me about the type annotation of {parameter}?",
                f"What's the type annotation for {parameter}?",
            ]
            function_parameter_types_answers = [
                f"Type annotation of {parameter} is '{parameter_annotation}'.",
                f"The type annotation of {parameter} is '{parameter_annotation}'.",
                f"The type annotation for {parameter} is '{parameter_annotation}'.",
            ]

            function_member_retrieval_chunks.append(function_parameter_types_context)
            function_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    function_parameter_types_context,
                    function_parameter_types_questions,
                    function_parameter_types_answers,
                )
            )

        if not (parameter_summary := function_parameter.parameter_summary):
            function_parameter_summary_context = f"{parameter} is not documented in the docstring."
            function_parameter_summary_questions = [
                f"What is {parameter} for?",
                f"Can you explain the purpose of {parameter}?",
                f"I'm not sure what {parameter} does. Can you help?",
                f"Could you clarify the role of {parameter}?",
                f"I'm confused about the {parameter}. What does it do?",
                f"What does {parameter} do?",
            ]
            function_parameter_summary_answers = [
                f"Docstring of {function_member} lacks a description for '{parameter_name}'.",
                f"The docstring of {function_member} doesn't provide a description.",
                f"Unfortunately, the docstring of {function_member} doesn't include"
                " a description.",
                f"The description is missing in the docstring of {function_member}.",
                f"The docstring of {function_member} doesn't contain a description.",
                f"There's no description in the docstring of {function_member}.",
            ]

            function_member_retrieval_chunks.append(function_parameter_summary_context)
            function_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    function_parameter_summary_context,
                    function_parameter_summary_questions,
                    function_parameter_summary_answers,
                )
            )
        else:
            function_parameter_summary_context = (
                f"In the docstring, {parameter} is described as '{parameter_summary}'."
            )
            function_parameter_summary_questions = [
                f"What is {parameter} for?",
                f"Can you explain the role of {parameter}?",
                f"I'm curious about the {parameter}. What does it do?",
                f"Could you tell me the purpose of {parameter}?",
                f"What's the function of {parameter}?",
                f"I'd like to know what '{parameter_name}' does in {function_member}.",
            ]
            function_parameter_summary_answers = [
                f"Based on {function_member} docstring, its role is '{parameter_summary}'.",
                f"According to the docstring of {function_member},"
                f"'{parameter_name}' is used for '{parameter_summary}'.",
                f"If you look at the docstring of {function_member}, you'll see that"
                f" '{parameter_name}' is responsible for '{parameter_summary}'.",
                f"The docstring of {function_member} indicates that"
                f" '{parameter_name}' serves the purpose of '{parameter_summary}'.",
                f"As per the docstring of {function_member}, '{parameter_name}' functions as:"
                f" '{parameter_summary}'.",
                f"The docstring of {function_member} states that"
                f" '{parameter_name}' does '{parameter_summary}'.",
            ]

            function_member_retrieval_chunks.append(function_parameter_summary_context)
            function_member_tuning_documents.extend(
                allocate_tuning_triplets(
                    function_parameter_summary_context,
                    function_parameter_summary_questions,
                    function_parameter_summary_answers,
                )
            )

    if (
        returns_annotation := member_type_details.function_returns.returns_annotation
    ) is EMPTY_SIGNATURE:
        function_return_type_context = (
            f"{function_member} has no return annotation, but its return can still be non-null."
        )
        function_return_type_questions = [
            f"What is the return type annotation of {function_member}?",
            f"Can you tell me the return type annotation of {function_member}?",
            f"I'm curious about return type annotation of {function_member}. What is it?",
            f"Do you know the return type annotation of {function_member}?",
            f"Could you inform me about the return type annotation of {function_member}?",
            f"What's the return type annotation for {function_member}?",
        ]
        function_return_type_answers = [
            f"{function_member} lacks a return type annotation. It may still return though.",
            f"The function {function_member} does not have a return type annotation."
            " However, it may still return.",
            f"{function_member} doesn't have a return type annotation."
            " But, it could still return.",
            f"Actually, {function_member} doesn't come with a return type annotation."
            " It's possible that it still returns though.",
            f"Sure, {function_member} is missing a return type annotation."
            " It might still return though.",
            f"It appears that {function_member} is without a return type annotation."
            " It may still have a return.",
        ]

        function_member_retrieval_chunks.append(function_return_type_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_return_type_context,
                function_return_type_questions,
                function_return_type_answers,
            )
        )
    else:
        function_return_type_context = (
            f"Return of {function_member} is annotated as '{returns_annotation}'."
        )
        function_return_type_questions = [
            f"What is the return type annotation of {function_member}?",
            f"Can you tell me the return type annotation of {function_member}?",
            f"I need to know the return type annotation of {function_member}.",
            f"Do you know the return type annotation of {function_member}?",
            f"Could you inform me about the return type annotation of {function_member}?",
            f"I'm curious about the return type annotation of {function_member}.",
        ]
        function_return_type_answers = [
            f"Return type annotation for {function_member} is '{returns_annotation}'.",
            f"The return type annotation for {function_member} is '{returns_annotation}'.",
            f"The return type for {function_member} is '{returns_annotation}'.",
        ]

        function_member_retrieval_chunks.append(function_return_type_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_return_type_context,
                function_return_type_questions,
                function_return_type_answers,
            )
        )

    if not (returns_summary := member_type_details.function_returns.returns_summary):
        function_return_summary_context = f"{function_member} does not document its return."
        function_return_summary_questions = [
            f"What does {function_member} return?",
            f"Can you tell me what {function_member} returns?",
            f"Do you know the return of {function_member}?",
            f"I'm curious about what {function_member} returns. Can you help?",
            f"What's the return of {function_member}?",
            f"Could you inform me about the return of {function_member}?",
        ]
        function_return_summary_answers = [
            f"Docstring of {function_member} does not describe its return.",
            f"Docstring of {function_member} doesn't provide information about its return.",
            f"Docstring of {function_member} doesn't specify what it returns.",
            f"The docstring of {function_member} doesn't clarify its return.",
            f"The return of {function_member} is not described in its docstring.",
            f"The docstring of {function_member} doesn't detail its return.",
        ]

        function_member_retrieval_chunks.append(function_return_summary_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_return_summary_context,
                function_return_summary_questions,
                function_return_summary_answers,
            )
        )
    else:
        function_return_summary_context = (
            f"Based on docstring, return of {function_member} is as follows: '{returns_summary}'."
        )
        function_return_summary_questions = [
            f"What does {function_member} return?",
            f"Can you tell me what {function_member} returns?",
            f"I'm curious about what {function_member} returns. Can you help?",
            f"Do you know what {function_member} returns?",
            f"I'd like to know what {function_member} returns.",
            f"Could you inform me about the return of {function_member}?",
        ]
        function_return_summary_answers = [
            f"Based on {function_member} docstring, the return contains: '{returns_summary}'.",
            f"As per docstring of {function_member}, it returns: '{returns_summary}'.",
            f"The docstring of {function_member} indicates that it returns: '{returns_summary}'.",
            f"The docstring of {function_member} states that it returns: '{returns_summary}'.",
            f"The docstring of {function_member} reveals that its return contains:"
            f" '{returns_summary}'.",
            f"The docstring of {function_member} specifies that it returns: '{returns_summary}'.",
        ]

        function_member_retrieval_chunks.append(function_return_summary_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_return_summary_context,
                function_return_summary_questions,
                function_return_summary_answers,
            )
        )

    if not (function_summary := member_type_details.function_summary):
        function_summary_context = f"Documentation for {function_member} is missing."
        function_summary_questions = [
            f"Summarise role of {function_member} in short.",
            f"Can you briefly explain the role of {function_member}?",
            f"What is the purpose of {function_member} as per its docstring?",
            f"Could you provide a summary of objective of {function_member}?",
            f"What does {function_member} do according to its docstring?",
        ]
        function_summary_answers = [
            f"{function_member} docstring lacks a summary of its objective.",
            f"The docstring of {function_member} doesn't provide its purpose.",
            f"The docstring of {function_member} doesn't clearly state its purpose.",
            f"The objective of {function_member} is not summarised in its docstring.",
            f"According to its docstring, role of {function_member} is not summarised.",
        ]

        function_member_retrieval_chunks.append(function_summary_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_summary_context, function_summary_questions, function_summary_answers
            )
        )
    else:
        function_summary_context = (
            f"{function_member} documents itself as follows: '{function_summary}'."
        )
        function_summary_questions = [
            f"Summarise role of {function_member} in short.",
            f"Can you briefly explain the role of {function_member}?",
            f"What does {function_member} do, in a nutshell?",
            f"Could you provide a short summary of role of {function_member}?",
            f"I need a brief explanation of what {function_member} does.",
            f"In brief, what is the role of {function_member}?",
        ]
        function_summary_answers = [
            f"Based on docstring, objective of {function_member} is to: '{function_summary}'.",
            f"According to the docstring, the purpose of {function_member} is:"
            f" '{function_summary}'.",
            f"In a nutshell, {function_member} is designed to: '{function_summary}'.",
            f"From docstring, {function_member} aims to: '{function_summary}'.",
            f"{function_member} is intended to: '{function_summary}'.",
            f"The role of {function_member} is to: '{function_summary}',"
            " according to the docstring.",
        ]

        function_member_retrieval_chunks.append(function_summary_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_summary_context, function_summary_questions, function_summary_answers
            )
        )

    if not (function_raises := member_type_details.function_raises):
        function_raise_types_context = (
            f"{function_member} does not document any specific exceptions in the docstring."
        )
        function_raise_types_questions = [
            f"Does {function_member} raise any specific exception?",
            f"Are there any specific exceptions that {function_member} raises?",
            f"Can you tell me if {function_member} raises any specific exceptions?",
            f"I want to know if {function_member} raises any specific exceptions."
            " Can you confirm?",
            f"Could {function_member} possibly raise any specific exceptions?",
            f"Is it possible for {function_member} to raise any specific exceptions?",
        ]
        function_raise_types_answers = [
            f"Docstring of {function_member} does not mention any specific exceptions.",
            f"No specific exceptions are mentioned in the docstring of {function_member}.",
            f"According to docstring, {function_member} does not raise exceptions.",
            f"Docstring of {function_member} does not mention exceptions.",
            f"The docstring of {function_member} does not indicate that"
            " it raises any specific exceptions.",
            f"The docstring of {function_member} does not suggest that"
            " it raises any specific exceptions.",
        ]

        function_member_retrieval_chunks.append(function_raise_types_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_raise_types_context,
                function_raise_types_questions,
                function_raise_types_answers,
            )
        )
    else:
        function_raise_types = enumerate_array_elements(
            function_raises, attribute="raises_details"
        )

        function_raise_types_context = (
            f"From docstring, {function_member} can raise the following: {function_raise_types}"
        )
        function_raise_types_questions = [
            f"Does {function_member} raise any specific exception?",
            f"Can you tell me if {function_member} raises any specific exceptions?",
            f"What exceptions, if any, does {function_member} raise?",
            f"I need to know if {function_member} throws any specific exceptions. Can you help?",
            f"Could you inform me about any specific exceptions that"
            f" {function_member} might raise?",
            f"I'm curious about the exceptions that {function_member} might throw."
            " Do you have any information?",
        ]
        function_raise_types_answers = [
            f"Based on docstring of {function_member}, it can raise the following:"
            f" {function_raise_types}.",
            f"According to docstring of {function_member}, it can raise these exceptions:"
            f" {function_raise_types}.",
            f"{function_member} can raise these exceptions as per its docstring:"
            f" {function_raise_types}.",
            f"{function_member} can throw following exceptions according to docstring:"
            f" {function_raise_types}.",
            f"The docstring of {function_member} indicates that"
            f" it can raise these exceptions: {function_raise_types}.",
            f"The docstring of {function_member} suggests that"
            f" it can throw the following exceptions: {function_raise_types}.",
        ]

        function_member_retrieval_chunks.append(function_raise_types_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_raise_types_context,
                function_raise_types_questions,
                function_raise_types_answers,
            )
        )

    if not (function_warns := member_type_details.function_warns):
        function_warn_types_context = (
            f"Mention of any warnings is missing in docstring of {function_member}."
        )
        function_warn_types_questions = [
            f"Does {function_member} throw any specific warnings?",
            f"Are there any specific warnings that {function_member} throws?",
            f"Can you tell me if {function_member} throws any specific warnings?",
            f"I want to know if {function_member} throws any specific warnings. Can you help?",
            f"Could you check if {function_member} throws any specific warnings?",
            f"Is it possible that {function_member} throws any specific warnings?",
        ]
        function_warn_types_answers = [
            f"Docstring of {function_member} lacks any mention of specific warnings.",
            f"There are no specific warnings mentioned in docstring of {function_member}.",
            f"According to the docstring of {function_member},"
            " it doesn't throw any specific warnings.",
            f"No mention of specific warnings are found in the docstring of {function_member}.",
            f"Based on the docstring of {function_member},"
            " it doesn't seem to throw any specific warnings.",
        ]

        function_member_retrieval_chunks.append(function_warn_types_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_warn_types_context,
                function_warn_types_questions,
                function_warn_types_answers,
            )
        )
    else:
        function_warn_types = enumerate_array_elements(function_warns, attribute="warns_details")

        function_warn_types_context = (
            f"{function_member} documents the following warnings: {function_warn_types}"
        )
        function_warn_types_questions = [
            f"Does {function_member} throw any specific warnings?",
            f"Can you tell me if {function_member} throws any specific warnings?",
            f"I'm curious, does {function_member} generate any particular warnings?",
            f"What specific warnings, if any, does {function_member} throw?",
            f"Could {function_member} possibly throw any specific warnings?",
            f"Are there any specific warnings that {function_member} throws?",
        ]
        function_warn_types_answers = [
            f"Based on the docstring, {function_member} can throw the following warnings:"
            f" {function_warn_types}.",
            f"According to docstring, {function_member} may throw these warnings:"
            f" {function_warn_types}.",
            f"Docstring indicates that {function_member} can generate these warnings:"
            f" {function_warn_types}.",
            f"{function_member} throws the following warnings as per the docstring:"
            f" {function_warn_types}.",
            f"Docstring of {function_member} mentions these specific warnings:"
            f" {function_warn_types}.",
            f"The docstring for {function_member} lists following warnings:"
            f" {function_warn_types}.",
        ]

        function_member_retrieval_chunks.append(function_warn_types_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_warn_types_context,
                function_warn_types_questions,
                function_warn_types_answers,
            )
        )

    if not (function_notes := member_type_details.function_notes):
        function_notes_context = f"{function_member} has no specific notes in the docstring."
        function_notes_questions = [
            f"Is there any specific details for {function_member} to be aware of?",
            f"Are there any particular details I should know about {function_member}?",
            f"What should I be aware of when using {function_member}?",
            f"Could you tell me if there are any specific details for {function_member}?",
            f"I'm curious if there are any specific details about {function_member}?",
            f"Do I need to be aware of any specific details for {function_member}?",
        ]
        function_notes_answers = [
            f"Docstring of {function_member} lacks any notes on specific details.",
            f"There are no specific details noted in the docstring of {function_member}.",
            f"The docstring of {function_member} does not contain any details to be aware of.",
            f"No specific details are mentioned in the docstring of {function_member}.",
            f"The docstring of {function_member} does not provide any specific details.",
            f"The docstring of {function_member} does not include any specific details.",
        ]

        function_member_retrieval_chunks.append(function_notes_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_notes_context, function_notes_questions, function_notes_answers
            )
        )
    else:
        function_notes_context = (
            f"Docstring for {function_member} has following notes: '{function_notes}'."
        )
        function_notes_questions = [
            f"Is there any specific details for {function_member} to be aware of?",
            f"What should I know about {function_member}?",
            f"Could you provide some details about {function_member}?",
            f"What are the important details of {function_member}?",
            f"Can you tell me more about {function_member}?",
            f"I need information about {function_member}.",
        ]
        function_notes_answers = [
            f"Docstring of {function_member} highlights the following: '{function_notes}'.",
            "Users should be aware that docstring includes the following details:"
            f" '{function_notes}'.",
            f"The docstring of {function_member} provides the following information:"
            f" '{function_notes}'.",
            f"The important details of {function_member} are highlighted in its docstring:"
            f" '{function_notes}'.",
            f"The docstring of {function_member} contains the following details:"
            f" '{function_notes}'.",
            f"The docstring of {function_member} contains the following information:"
            f" '{function_notes}'.",
        ]

        function_member_retrieval_chunks.append(function_notes_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_notes_context, function_notes_questions, function_notes_answers
            )
        )

    if not (function_references := member_type_details.function_references):
        function_references_context = (
            f"{function_member} documents no references in its docstring."
        )
        function_references_questions = [
            f"Is there any reference for {function_member}?",
            f"Can I find any references in the documentation for {function_member}?",
            f"Does the documentation for {function_member} include any references?",
            f"Are there references available in the {function_member} documentation?",
            f"I'm looking for references in {function_member} documentation. Are there any?",
            f"Could you tell me if there are any references for {function_member}?",
        ]
        function_references_answers = [
            f"Documentation for {function_member} contains no references.",
            f"The documentation for {function_member} does not contain any references.",
            f"There are no references in the documentation for {function_member}.",
            f"The {function_member} documentation does not include any references.",
            f"The documentation for {function_member} contains no references.",
            f"Documentation for {function_member} lacks any references.",
        ]

        function_member_retrieval_chunks.append(function_references_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_references_context,
                function_references_questions,
                function_references_answers,
            )
        )
    else:
        function_references_context = (
            f"{function_member} list the following references: {function_references}"
        )
        function_references_questions = [
            f"Is there any reference for {function_member}?",
            f"Can you provide a reference for {function_member}?",
            f"Where can I find a reference for {function_member}?",
            f"Could you point me to the reference for {function_member}?",
            f"I'm looking for a reference for {function_member}. Can you help?",
            f"What's the reference for {function_member}?",
        ]
        function_references_answers = [
            f"The docstring links the following: '{function_references}'.",
            f"The docstring provides the following reference: '{function_references}'.",
            f"The docstring links to: '{function_references}'.",
            f"The docstring points to these reference: '{function_references}'.",
            f"The docstring links to this reference: '{function_references}'.",
            f"The reference for that is in the docstring: '{function_references}'.",
        ]

        function_member_retrieval_chunks.append(function_references_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_references_context,
                function_references_questions,
                function_references_answers,
            )
        )

    if not (function_examples := member_type_details.function_examples):
        function_examples_context = f"Documentation of {function_member} lacks any examples."
        function_examples_questions = [
            f"Is there any example for {function_member}?",
            f"Can I find an example for {function_member} in the docstring?",
            f"Does the docstring for {function_member} include any examples?",
            f"I'm looking for an example of {function_member} in docstring, is there one?",
            f"Are there any examples provided in the docstring for {function_member}?",
            f"Could you tell me if there's an example for {function_member} in docstring?",
        ]
        function_examples_answers = [
            f"Docstring for {function_member} lacks any examples.",
            f"Docstring for {function_member} does not contain any examples.",
            f"The docstring for {function_member} does not include any examples.",
            f"Docstring for {function_member} does not provide any examples.",
            f"No examples are provided in the docstring for {function_member}.",
            f"{function_member} documents no examples.",
        ]

        function_member_retrieval_chunks.append(function_examples_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_examples_context, function_examples_questions, function_examples_answers
            )
        )
    else:
        function_examples_context = (
            f"Docstring of {function_member} contains following examples: '{function_examples}'."
        )
        function_examples_questions = [
            f"Is there any example for {function_member}?",
            f"Can you provide an example of {function_member}?",
            f"I'm looking for examples of {function_member}, can you help?",
            f"Where can I find examples for {function_member}?",
            f"Could you show me some examples of {function_member}?",
            f"I need examples for {function_member}, where can I find them?",
        ]
        function_examples_answers = [
            f"Documentation of {function_member} contains these examples: '{function_examples}'.",
            f"In documentation of {function_member}, these examples can be found:"
            f" '{function_examples}'.",
            f"Examples for {function_member} are available in its documentation:"
            f" '{function_examples}'.",
            f"In documentation for {function_member}, these examples can be found:"
            f" '{function_examples}'.",
            f"The documentation of {function_member} includes these examples:"
            f" '{function_examples}'.",
        ]

        function_member_retrieval_chunks.append(function_examples_context)
        function_member_tuning_documents.extend(
            allocate_tuning_triplets(
                function_examples_context, function_examples_questions, function_examples_answers
            )
        )

    function_member_dataset = Dataset(
        retrieval_chunks=function_member_retrieval_chunks[:2],
        tuning_documents=function_member_tuning_documents,
    )

    return function_member_dataset, function_member_retrieval_chunks


@pydantic.validate_call(validate_return=True)
def generate_member_dataset(member_details: MemberDetails) -> tuple[Dataset, ...]:  # noqa: PLR0915
    """Create a dataset for a member.

    Parameters
    ----------
    member_details : MemberDetails
        all details of the member

    Returns
    -------
    tuple[Dataset, ...]
        all documents for retrieval and tuning for querying member documentation

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
    member_tuning_documents: list[Document] = []

    module_parent_context = f"{member} is part of parent module {member_details.member_module}."
    module_parent_questions = [
        f"What is the parent module of {member}?",
        f"Can you tell me the parent module of {member}?",
        f"I'm trying to find the parent module of {member}, can you help?",
        f"Do you know the parent module of {member}?",
        f"I need to know the parent module of {member}, can you provide that?",
        f"Could you inform me about the parent module of {member}?",
    ]
    module_parent_answers = [
        f"'{member_details.member_module}' is the name of its parent module.",
        f"The parent module of {member} is '{member_details.member_module}'.",
        f"Parent module of {member} is '{member_details.member_module}'.",
        f"'{member_details.member_module}' is parent module of {member}.",
    ]

    member_retrieval_chunks.append(module_parent_context)
    member_tuning_documents.extend(
        allocate_tuning_triplets(
            module_parent_context, module_parent_questions, module_parent_answers
        )
    )

    member_full_name_context = f"Full name of {member} is '{member_full_name}'."
    member_full_name_questions = [
        f"What is the full name of {member}?",
        f"Can you tell me the full name of the {member}?",
        f"I need to know the full name of {member}. Can you help?",
        f"What's the fully qualified name for the {member}?",
        f"Could you provide the full name of the {member}?",
        f"I'm looking for the full name of {member}. What is it?",
    ]
    member_full_name_answers = [
        f"'{member_full_name}' is its fully qualified name.",
        f"The fully qualified name of {member} is '{member_full_name}'.",
        f"The full name of {member} is '{member_full_name}'.",
        f"The fully qualified name for {member} is '{member_full_name}'.",
        f"The full name of the {member} is '{member_full_name}'.",
    ]

    member_retrieval_chunks.append(member_full_name_context)
    member_tuning_documents.extend(
        allocate_tuning_triplets(
            member_full_name_context, member_full_name_questions, member_full_name_answers
        )
    )

    member_hierarchy = enumerate_array_elements(member_details.member_hierarchy)

    member_hierarchy_context = f"Hierarchy of {member} is as follows: {member_hierarchy}."
    member_hierarchy_questions = [
        f"What is the hierarchy of {member}?",
        f"Can you explain the hierarchy of the {member}?",
        f"Could you tell me the hierarchy of {member}?",
        f"I would like to know the hierarchy of {member}. Can you provide that?",
        f"Please provide the hierarchy of {member}.",
        f"I'm interested in the hierarchy of {member}. Could you share it?",
    ]
    member_hierarchy_answers = [
        f"The hierarchy of {member} is as follows: {member_hierarchy}.",
        f"The hierarchy of the {member} is: {member_hierarchy}.",
        f"The hierarchy of {member} is: {member_hierarchy}.",
    ]

    member_retrieval_chunks.append(member_hierarchy_context)
    member_tuning_documents.extend(
        allocate_tuning_triplets(
            member_hierarchy_context, member_hierarchy_questions, member_hierarchy_answers
        )
    )

    if not (member_docstring := member_details.member_docstring):
        member_documentation_context = (
            f"Unfortunately, {member} currently does not have any documentation."
        )
        member_documentation_questions = [
            f"What is the documentation of {member}?",
            f"Can you provide the documentation for the {member}?",
            f"Is there any documentation available for the {member}?",
            f"Could you show me the documentation of the {member}?",
            f"I'm looking for the documentation of {member}. Can you help?",
        ]
        member_documentation_answers = [
            f"{member} does not have any documentation.",
            f"The {member} does not have any documentation.",
            f"There is no documentation available for the {member}.",
        ]

        member_retrieval_chunks.append(member_documentation_context)
        member_tuning_documents.extend(
            allocate_tuning_triplets(
                member_documentation_context,
                member_documentation_questions,
                member_documentation_answers,
            )
        )
    else:
        member_documentation_context = (
            f"The following is the documentation of {member}: '{member_docstring}'."
        )
        member_documentation_questions = [
            f"What does {member} do?",
            f"Can you explain the function of the {member}?",
            f"I'm not sure what {member} does. Can you clarify?",
            f"Could you tell me about the {member}?",
            f"I need information on the {member}.",
            f"What's the purpose of the {member}?",
        ]
        member_documentation_answers = [
            f"Its documentation is as follows: '{member_docstring}'.",
            f"Here is its documentation: '{member_docstring}'.",
            f"Here's its documentation for clarification: '{member_docstring}'.",
            f"Its documentation is: '{member_docstring}'.",
            f"Here's the documentation you need: '{member_docstring}'.",
            f"The purpose is described in its documentation: '{member_docstring}'.",
        ]

        member_retrieval_chunks.append(member_documentation_context)
        member_tuning_documents.extend(
            allocate_tuning_triplets(
                member_documentation_context,
                member_documentation_questions,
                member_documentation_answers,
            )
        )

    if (member_type_details := member_details.member_type_details) is not None:
        member_type = member_type_details.member_type

        member_type_context = f"'{member_name}' is a Python {member_type.value}."
        member_type_questions = [
            f"What is the type of {member}?",
            f"Can you tell me the type of the {member}?",
            f"I would like to know the type of {member}. Can you help?",
            f"Do you know the type of {member}?",
            f"Could you inform me about the type of {member}?",
            f"I'm curious about type of {member}. Can you provide some information?",
        ]
        member_type_answers = [
            f"{member} is of '{member_type.value}' type.",
            f"The {member} is of '{member_type.value}' type.",
        ]

        member_retrieval_chunks.append(member_type_context)
        member_tuning_documents.extend(
            allocate_tuning_triplets(
                member_type_context, member_type_questions, member_type_answers
            )
        )

    if member_type_details is None:
        member_retrieval_chunks.insert(0, f"'{member_name}' is a Python object.")

        member_dataset = Dataset(
            retrieval_chunks=member_retrieval_chunks, tuning_documents=member_tuning_documents
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
        tuning_documents=member_tuning_documents,
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
