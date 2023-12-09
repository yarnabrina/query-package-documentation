import inspect
import random

import pydantic

from .utils_generation import (
    ClassDetails,
    Document,
    EnumDetails,
    FunctionDetails,
    MemberDetails,
    MemberType,
    Module,
    Package,
)

random.seed(a=0)


@pydantic.validate_call(validate_return=True)
def generate_dataset(question_answer_pairs: list[tuple[str, str]]) -> list[Document]:
    dataset = [
        Document(question=question, answer=answer)
        for question, answer in random.sample(question_answer_pairs, 3)
    ]

    return dataset


@pydantic.validate_call(validate_return=True)
def generate_package_dataset(package_contents: Package) -> list[Document]:
    package_name = package_contents.package_name
    package_full_name = package_contents.package_qualified_name

    package_dataset: list[Document] = []

    if (parent_package := package_contents.parent_package_name) is None:
        root_package_question_answer_pairs = [
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
        package_dataset.extend(generate_dataset(root_package_question_answer_pairs))

        parent_package_question_answer_pairs = [
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
        package_dataset.extend(generate_dataset(parent_package_question_answer_pairs))
    else:
        parent_package_question_answer_pairs = [
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
        package_dataset.extend(generate_dataset(parent_package_question_answer_pairs))

        package_full_name_question_answer_pairs = [
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
        package_dataset.extend(generate_dataset(package_full_name_question_answer_pairs))

        package_hierarchy = " ".join(
            f"{counter + 1}. {node}"
            for counter, node in enumerate(package_contents.package_hierarchy)
        )
        package_hierarchy_question_answer_pairs = [
            (
                f"What is the hierarchy of {package_name} package?",
                f"The hierarchy of '{package_name}' package is as follows: {package_hierarchy}.",
            ),
            (
                f"Can you explain the hierarchy of the {package_name} package?",
                f"Sure, the hierarchy of the '{package_name}' package is: {package_hierarchy}.",
            ),
            (
                f"Could you describe the structure of the {package_name} package?",
                f"Of course, the structure of '{package_name}' package is: {package_hierarchy}.",
            ),
            (
                f"I need to understand the hierarchy of the {package_name} package. Can you help?",
                f"Absolutely, the hierarchy of '{package_name}' package is: {package_hierarchy}.",
            ),
            (
                f"Please provide the hierarchy of the {package_name} package.",
                f"The hierarchy of the '{package_name}' package is: {package_hierarchy}.",
            ),
            (
                f"I'm interested in the structure of the {package_name} package. What is it?",
                f"The structure of '{package_name}' package is as follows: {package_hierarchy}.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_hierarchy_question_answer_pairs))

    if not (children_sub_packages := package_contents.children_sub_packages_names):
        package_sub_package_question_answer_pairs = [
            (
                f"List the sub-packages of '{package_full_name}' package.",
                f"'{package_full_name}' package does not have any further sub-packages.",
            ),
            (
                f"What are the sub-packages of the '{package_full_name}' package?",
                f"The '{package_full_name}' package does not contain any sub-packages.",
            ),
            (
                f"Could you tell me the sub-packages of '{package_full_name}' package?",
                f"I'm sorry, but the '{package_full_name}' package doesn't have any sub-packages.",
            ),
            (
                f"I need to know the sub-packages of '{package_full_name}' package."
                " Can you list them?",
                f"Unfortunately, '{package_full_name}' package doesn't include any sub-packages.",
            ),
            (
                f"Can you provide a list of sub-packages for the '{package_full_name}' package?",
                f"There are no sub-packages in the '{package_full_name}' package.",
            ),
            (
                f"Identify the sub-packages of '{package_full_name}' package.",
                f"No sub-packages are present in the '{package_full_name}' package.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_sub_package_question_answer_pairs))
    else:
        package_sub_packages = " ".join(
            f"{counter + 1}. {sub_package}"
            for counter, sub_package in enumerate(children_sub_packages)
        )
        package_sub_package_question_answer_pairs = [
            (
                f"List the sub-packages of '{package_full_name}' package.",
                f"Sub-packages of '{package_full_name}' package"
                f" are as follows: {package_sub_packages}.",
            ),
            (
                f"What are the sub-packages of the '{package_full_name}' package?",
                f"The '{package_full_name}' package has "
                f"the following sub-packages: {package_sub_packages}.",
            ),
            (
                f"Could you tell me the sub-packages of '{package_full_name}' package?",
                f"Sure, the sub-packages of '{package_full_name}' package"
                f" are: {package_sub_packages}.",
            ),
            (
                f"I need to know the sub-packages of '{package_full_name}' package."
                " Can you list them?",
                f"Of course, the sub-packages of '{package_full_name}' package"
                " are: {package_sub_packages}.",
            ),
            (
                f"Please provide the sub-packages of '{package_full_name}' package.",
                f"The sub-packages of '{package_full_name}' package are: {package_sub_packages}.",
            ),
            (
                f"Can you enumerate the sub-packages of '{package_full_name}' package?",
                f"Certainly, the sub-packages of '{package_full_name}' package"
                " are: {package_sub_packages}.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_sub_package_question_answer_pairs))

    if not (children_modules := package_contents.children_modules_names):
        package_module_question_answer_pairs = [
            (
                f"What are the modules of '{package_full_name}' package?",
                f"'{package_full_name}' does not have any direct modules under itself.",
            ),
            (
                f"Can you list the modules under the '{package_full_name}' package?",
                f"There are no direct modules under the '{package_full_name}' package.",
            ),
            (
                f"Does the '{package_full_name}' package contain any modules?",
                f"No, the '{package_full_name}' package does not contain any direct modules.",
            ),
            (
                f"I'm looking for the modules of '{package_full_name}' package. Can you help?",
                f"I'm sorry, but '{package_full_name}' package does not have any direct modules.",
            ),
            (
                f"Tell me about the modules of '{package_full_name}' package.",
                f"Actually, the '{package_full_name}' package does not have any direct modules.",
            ),
            (
                f"Are there any modules under the '{package_full_name}' package?",
                f"No, there aren't any direct modules under the '{package_full_name}' package.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_module_question_answer_pairs))
    else:
        package_modules = " ".join(
            f"{counter + 1}. {sub_package}" for counter, sub_package in enumerate(children_modules)
        )
        package_module_question_answer_pairs = [
            (
                f"What are the modules of '{package_full_name}' package?",
                f"Direct modules under '{package_full_name}' are as follows: {package_modules}.",
            ),
            (
                f"Can you list the modules of the '{package_full_name}' package?",
                f"Sure, the direct modules under '{package_full_name}' are: {package_modules}.",
            ),
            (
                f"I need to know the modules of the '{package_full_name}' package.",
                f"The modules you're looking for in '{package_full_name}' are: {package_modules}.",
            ),
            (
                f"Could you tell me what the modules of the '{package_full_name}' package are?",
                f"Of course, the modules under '{package_full_name}' are: {package_modules}.",
            ),
            (
                f"I'm interested in the modules of the '{package_full_name}' package.",
                f"The modules in '{package_full_name}' are: {package_modules}.",
            ),
            (
                f"What modules does the '{package_full_name}' package contain?",
                f"The '{package_full_name}' package contains these modules: {package_modules}.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_module_question_answer_pairs))

    if not (package_summary := package_contents.package_summary):
        package_summary_question_answer_pairs = [
            (
                f"What does '{package_full_name}' package do?",
                f"'{package_full_name}' package does not have any documentation.",
            ),
            (
                f"Can you tell me the functionality of the '{package_full_name}' package?",
                f"Unfortunately, the '{package_full_name}' package provides no documentation.",
            ),
            (
                f"I'm curious about what the '{package_full_name}' package does."
                " Can you enlighten me?",
                f"I'm sorry, but the '{package_full_name}' package"
                " does not come with any documentation.",
            ),
            (
                f"Could you explain the purpose of the '{package_full_name}' package?",
                f"Regrettably, the '{package_full_name}' package lacks any form of documentation.",
            ),
            (
                f"What's the role of the '{package_full_name}' package?",
                f"The '{package_full_name}' package does not offer any documentation.",
            ),
            (
                f"What functionality does the '{package_full_name}' package provide?",
                f"The '{package_full_name}' package does not have any available documentation.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_summary_question_answer_pairs))
    else:
        package_summary_question_answer_pairs = [
            (
                f"What does '{package_full_name}' package do?",
                f"Its documentation is as follows: '{package_summary}'.",
            ),
            (
                f"Can you tell me about the '{package_full_name}' package?",
                f"Sure, here is its documentation: '{package_summary}'.",
            ),
            (
                f"I'd like to know what the '{package_full_name}' package does.",
                f"Of course, here's the documentation for it: '{package_summary}'.",
            ),
            (
                f"Could you explain the functionality of the '{package_full_name}' package?",
                f"Absolutely, the documentation states: '{package_summary}'.",
            ),
            (
                f"What's the purpose of the '{package_full_name}' package?",
                f"The purpose is described in its documentation: '{package_summary}'.",
            ),
            (
                f"I'm curious about the '{package_full_name}' package, what does it do?",
                f"Good question, its documentation reads: '{package_summary}'.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_summary_question_answer_pairs))

    if not (package_exports := package_contents.package_all_exports):
        package_members_question_answer_pairs = [
            (
                f"What are the public members of the '{package_full_name}' package?",
                f"'{package_full_name}' package does not have"
                " any public member exported through '__all__'.",
            ),
            (
                f"Can you list the public members of the '{package_full_name}' package?",
                f"The '{package_full_name}' package does not export"
                " any public members through '__all__'.",
            ),
            (
                f"Are there any public members in the '{package_full_name}' package?",
                f"No, the '{package_full_name}' package does not have"
                " any public members exported through '__all__'.",
            ),
            (
                f"I'm looking for public members of '{package_full_name}' package. Can you help?",
                f"Sure, but the '{package_full_name}' package does not have"
                " any public members exported through '__all__'.",
            ),
            (
                f"Could you tell me the public members of the '{package_full_name}' package?",
                f"Unfortunately, the '{package_full_name}' package does not have"
                " any public members exported through '__all__'.",
            ),
            (
                f"I'd like to know the public members of the '{package_full_name}' package."
                " Can you provide that information?",
                f"I'm sorry, but the '{package_full_name}' package does not have"
                " any public members exported through '__all__'.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_members_question_answer_pairs))
    else:
        package_public_members = " ".join(
            f"{counter + 1}. {package_export}"
            for counter, package_export in enumerate(package_exports)
        )
        package_members_question_answer_pairs = [
            (
                f"What are the public members of the '{package_full_name}' package?",
                f"'{package_full_name}' package publicly exports"
                f" the following members using '__all__': {package_public_members}.",
            ),
            (
                f"Can you list the public members of the '{package_full_name}' package?",
                f"Sure, the '{package_full_name}' package publicly exports"
                f" these members using '__all__': {package_public_members}.",
            ),
            (
                f"I need to know the public members of the '{package_full_name}' package."
                " Can you tell me?",
                f"Of course, the '{package_full_name}' package publicly exports"
                f" these members using '__all__': {package_public_members}.",
            ),
            (
                f"Could you tell me what the '{package_full_name}' package publicly exports?",
                f"The '{package_full_name}' package publicly exports"
                f" the following members using '__all__': {package_public_members}.",
            ),
            (
                f"I'm interested in the public members of the '{package_full_name}' package."
                " What are they?",
                f"The '{package_full_name}' package publicly exports"
                f" these members using '__all__': {package_public_members}.",
            ),
        ]
        package_dataset.extend(generate_dataset(package_members_question_answer_pairs))

    return package_dataset


@pydantic.validate_call(validate_return=True)
def generate_module_dataset(module_members: Module) -> list[Document]:
    module_name = module_members.module_name
    module_full_name = module_members.module_qualified_name

    module_dataset: list[Document] = []

    module_package_question_answer_pairs = [
        (
            f"Can you tell the the parent package of '{module_name}' module?",
            f"'{module_members.package_name}' is the parent package of '{module_name}'.",
        ),
        (
            f"What is the parent package of the '{module_name}' module?",
            f"The parent package of '{module_name}' module is '{module_members.package_name}'.",
        ),
        (
            f"I'm trying to find the parent package of the '{module_name}' module. Can you help?",
            f"Sure, parent package of '{module_name}' module is '{module_members.package_name}'.",
        ),
        (
            f"Could you inform me about the parent package of the '{module_name}' module?",
            f"Certainly, '{module_members.package_name}' is the"
            f" parent package of the '{module_name}' module.",
        ),
        (
            f"I need to know the parent package of the '{module_name}' module."
            " Can you provide that information?",
            f"Absolutely, the parent package of the '{module_name}' module"
            f" is '{module_members.package_name}'.",
        ),
        (
            f"Can you identify the parent package for the '{module_name}' module?",
            f"Yes, parent package for '{module_name}' module is '{module_members.package_name}'.",
        ),
    ]
    module_dataset.extend(generate_dataset(module_package_question_answer_pairs))

    module_full_name_question_answer_pairs = [
        (
            f"Specify the full name of '{module_name}' module?",
            f"'{module_full_name}' is fully qualified name for '{module_name}' module.",
        ),
        (
            f"What is the fully qualified name for the '{module_name}' module?",
            f"The fully qualified name for the '{module_name}' module is '{module_full_name}'.",
        ),
        (
            f"Could you tell me the full name of the '{module_name}' module?",
            f"Sure, the full name of the '{module_name}' module is '{module_full_name}'.",
        ),
        (
            f"I need the full name of the '{module_name}' module. Can you provide it?",
            f"Of course, the full name of the '{module_name}' module is '{module_full_name}'.",
        ),
        (
            f"Can you specify the fully qualified name of the '{module_name}' module?",
            f"Yes, fully qualified name of the '{module_name}' module is '{module_full_name}'.",
        ),
        (
            f"I'm looking for the full name of the '{module_name}' module. What is it?",
            f"Full name of the '{module_name}' module you're looking for is '{module_full_name}'.",
        ),
    ]
    module_dataset.extend(generate_dataset(module_full_name_question_answer_pairs))

    module_hierarchy = " ".join(
        f"{counter + 1}. {node}" for counter, node in enumerate(module_members.module_hierarchy)
    )
    module_hierarchy_question_answer_pairs = [
        (
            f"What is the hierarchy of {module_name} module?",
            f"The hierarchy of '{module_name}' module is as follows: {module_hierarchy}.",
        ),
        (
            f"Can you explain the hierarchy of the {module_name} module?",
            f"Sure, the hierarchy of the '{module_name}' module is: {module_hierarchy}.",
        ),
        (
            f"Could you describe the structure of the {module_name} module?",
            f"Of course, the structure of the '{module_name}' module is: {module_hierarchy}.",
        ),
        (
            f"I need to understand the hierarchy of the {module_name} module. Can you help?",
            f"Absolutely, the hierarchy of the '{module_name}' module is: {module_hierarchy}.",
        ),
        (
            f"Please provide the hierarchy of the {module_name} module.",
            f"The hierarchy of the '{module_name}' module is: {module_hierarchy}.",
        ),
        (
            f"What does the hierarchy of the {module_name} module look like?",
            f"The hierarchy of the '{module_name}' module looks like this: {module_hierarchy}.",
        ),
    ]
    module_dataset.extend(generate_dataset(module_hierarchy_question_answer_pairs))

    module_member_names = " ".join(
        f"{counter + 1}. {member.member_name}"
        for counter, member in enumerate(module_members.module_members)
    )
    module_members_question_answer_pairs = [
        (
            f"List the members of '{module_name}' module.",
            f"Members of '{module_full_name}' are as follows: {module_member_names}.",
        ),
        (
            f"What are the members of the '{module_name}' module?",
            f"The '{module_full_name}' module has the following members: {module_member_names}.",
        ),
        (
            f"Can you tell me the members of the '{module_name}' module?",
            f"Sure, the members of the '{module_full_name}' module are: {module_member_names}.",
        ),
        (
            f"I need to know the members of the '{module_name}' module.",
            f"Members of '{module_full_name}' module you asked for are: {module_member_names}.",
        ),
        (
            f"Could you list the members of the '{module_name}' module?",
            f"Of course, members of the '{module_full_name}' module are: {module_member_names}.",
        ),
        (
            f"Please provide the members of the '{module_name}' module.",
            f"Members of '{module_full_name}' module you requested are: {module_member_names}.",
        ),
    ]
    module_dataset.extend(generate_dataset(module_members_question_answer_pairs))

    if not (module_summary := module_members.module_summary):
        module_summary_question_answer_pairs = [
            (
                f"What is the '{module_full_name}' module for?",
                f"'{module_name}' member does not have any documentation.",
            ),
            (
                f"Can you tell me the purpose of the '{module_full_name}' module?",
                f"The '{module_name}' member lacks any documentation.",
            ),
            (
                f"I'd like to know what the '{module_full_name}' module is used for.",
                f"Unfortunately, there is no documentation for the '{module_name}' member.",
            ),
            (
                f"Could you explain the function of the '{module_full_name}' module?",
                f"Regrettably, the '{module_name}' member doesn't come with any documentation.",
            ),
            (
                f"What does the '{module_full_name}' module do?",
                f"The '{module_name}' member is without any documentation.",
            ),
        ]
        module_dataset.extend(generate_dataset(module_summary_question_answer_pairs))
    else:
        module_summary_question_answer_pairs = [
            (
                f"What is the '{module_name}' module for?",
                f"'{module_full_name}' module documents itself as follows: '{module_summary}'.",
            ),
            (
                f"Can you tell me the purpose of the '{module_name}' module?",
                f"Purpose of '{module_full_name}' module is documented as: '{module_summary}'.",
            ),
            (
                f"I'm curious about the '{module_name}' module. What does it do?",
                f"The '{module_full_name}' module is described as: '{module_summary}'.",
            ),
            (
                f"Could you explain the functionality of the '{module_name}' module?",
                f"The functionality of the '{module_full_name}' module is"
                f" described as: '{module_summary}'.",
            ),
            (
                f"I'd like to know more about the '{module_name}' module. What's its role?",
                f"The role of the '{module_full_name}' module is: '{module_summary}'.",
            ),
            (
                f"What's the use of the '{module_name}' module?",
                f"Use of the '{module_full_name}' module is documented as: '{module_summary}'.",
            ),
        ]
        module_dataset.extend(generate_dataset(module_summary_question_answer_pairs))

    if not (module_exports := module_members.module_all_exports):
        module_exports_question_answer_pairs = [
            (
                f"Tell me the public members of the '{module_full_name}' module.",
                f"'{module_name}' module lacks any public member exported through '__all__'.",
            ),
            (
                f"What are the public members of the '{module_full_name}' module?",
                "There are no public members exported through '__all__'"
                f" in the '{module_name}' module.",
            ),
            (
                f"Could you list the public members of the '{module_full_name}' module?",
                f"Unfortunately, the '{module_name}' module does not export"
                " any public members through '__all__'.",
            ),
            (
                f"I need to know the public members of the '{module_full_name}' module.",
                f"The '{module_name}' module does not have any public members"
                " exported through '__all__'.",
            ),
            (
                f"Can you show me the public members of the '{module_full_name}' module?",
                f"The '{module_name}' module does not contain any public members"
                " exported through '__all__'.",
            ),
            (
                f"I'm interested in the public members of the '{module_full_name}' module."
                " What are they?",
                f"'{module_name}' module does not export any public members through '__all__'.",
            ),
        ]
        module_dataset.extend(generate_dataset(module_exports_question_answer_pairs))
    else:
        module_public_exports = " ".join(
            f"{counter + 1}. {module_export}"
            for counter, module_export in enumerate(module_exports)
        )
        module_exports_question_answer_pairs = [
            (
                f"Tell me the public members of the '{module_name}' module.",
                f"{module_full_name} publicly exports the following members"
                f" using '__all__': {module_public_exports}.",
            ),
            (
                f"What are the public members of the '{module_name}' module?",
                f"The '{module_name}' module publicly exports the following members"
                f" using '__all__': {module_public_exports}.",
            ),
            (
                f"Could you list the public members of the '{module_name}' module?",
                f"Sure, the '{module_name}' module publicly exports these members"
                f" using '__all__': {module_public_exports}.",
            ),
            (
                f"I need to know the public members of the '{module_name}' module.",
                f"The '{module_name}' module publicly exports these members"
                f" using '__all__': {module_public_exports}.",
            ),
            (
                f"Can you show me the public members of the '{module_name}' module?",
                f"Of course, the '{module_name}' module publicly exports the following members"
                f" using '__all__': {module_public_exports}.",
            ),
        ]
        module_dataset.extend(generate_dataset(module_exports_question_answer_pairs))

    return module_dataset


@pydantic.validate_call(validate_return=True)
def generate_enum_member_dataset(
    enum_member: str, member_type_details: EnumDetails
) -> list[Document]:
    enum_member_dataset: list[Document] = []

    enum_member_count = len(member_type_details.enum_members)
    enum_member_count_question_answer_pairs = [
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
    enum_member_dataset.extend(generate_dataset(enum_member_count_question_answer_pairs))

    enum_members = " ".join(
        f"{counter + 1}. {enum_member.enum_member}"
        for counter, enum_member in enumerate(member_type_details.enum_members)
    )
    enum_members_question_answer_pairs = [
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
    enum_member_dataset.extend(generate_dataset(enum_members_question_answer_pairs))

    enum_member_names = " ".join(
        f"{counter + 1}. {enum_member.enum_member_name}"
        for counter, enum_member in enumerate(member_type_details.enum_members)
    )
    enum_member_names_question_answer_pairs = [
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
            f"Of course, different members of {enum_member} have"
            f" these names: {enum_member_names}.",
        ),
        (
            f"Show me the names of different members of {enum_member}.",
            f"The names of different members of {enum_member} are: {enum_member_names}.",
        ),
    ]
    enum_member_dataset.extend(generate_dataset(enum_member_names_question_answer_pairs))

    enum_member_values = " ".join(
        f"{counter + 1}. {enum_member.enum_member_value}"
        for counter, enum_member in enumerate(member_type_details.enum_members)
    )
    enum_member_values_question_answer_pairs = [
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
    enum_member_dataset.extend(generate_dataset(enum_member_values_question_answer_pairs))

    return enum_member_dataset


@pydantic.validate_call(validate_return=True)
def generate_class_member_dataset(  # noqa: C901, PLR0912, PLR0915
    class_member: str, member_type_details: ClassDetails
) -> list[Document]:
    class_member_dataset: list[Document] = []

    if not (class_parameters := member_type_details.class_parameters):
        class_parameters_question_answer_pairs = [
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
        class_member_dataset.extend(generate_dataset(class_parameters_question_answer_pairs))
    else:
        class_parameter_names = " ".join(
            f"{counter + 1}. {class_parameter.parameter_details}"
            for counter, class_parameter in enumerate(class_parameters)
        )
        class_parameters_question_answer_pairs = [
            (
                f"What are the different parameters of {class_member}?",
                f"{class_member} supports these arguments to initiate"
                f" a new instance: {class_parameter_names}.",
            ),
            (
                f"Can you list the parameters for {class_member}?",
                f"Sure, {class_member} can be initiated with"
                f" these arguments: {class_parameter_names}.",
            ),
            (
                f"I need to know the parameters of {class_member}.",
                f"The parameters to initiate a new instance"
                f" of {class_member} are: {class_parameter_names}.",
            ),
            (
                f"Tell me the parameters that {class_member} supports.",
                f"{class_member} can be initiated with these arguments: {class_parameter_names}.",
            ),
            (
                f"What arguments does {class_member} take for initialization?",
                f"To initialize {class_member}, you can use"
                f" these arguments: {class_parameter_names}.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_parameters_question_answer_pairs))

    for class_parameter in class_parameters:
        parameter_name = class_parameter.parameter_name

        if (parameter_default := class_parameter.parameter_default) is inspect._empty:
            class_parameter_defaults_question_answer_pairs = [
                (
                    f"Tell default value of '{parameter_name}' in {class_member}.",
                    f"'{parameter_name}' argument does not have a default value.",
                ),
                (
                    f"What is the default value of '{parameter_name}' in {class_member}?",
                    f"The '{parameter_name}' argument does not have a default value.",
                ),
                (
                    f"Could you inform me about the default value of '{parameter_name}'"
                    f" in {class_member}?",
                    f"Sure, the '{parameter_name}' argument does not have a default value.",
                ),
                (
                    f"I need to know the default value of '{parameter_name}' in {class_member}."
                    " Can you help?",
                    f"Of course, the '{parameter_name}' argument does not have a default value.",
                ),
                (
                    f"Can you tell me if '{parameter_name}' in {class_member} has default value?",
                    f"No, the '{parameter_name}' argument does not have a default value.",
                ),
                (
                    f"I'm curious about default value of '{parameter_name}' in {class_member}.",
                    f"Well, the '{parameter_name}' argument does not have a default value.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_parameter_defaults_question_answer_pairs)
            )
        else:
            class_parameter_defaults_question_answer_pairs = [
                (
                    f"Tell default value of '{parameter_name}' in {class_member}.",
                    f"Argument '{parameter_name}' takes {parameter_default} value by default.",
                ),
                (
                    f"What is the default value of '{parameter_name}' in {class_member}?",
                    f"The default value of '{parameter_name}' in {class_member}"
                    f" is {parameter_default}.",
                ),
                (
                    f"Could you inform me about the default value of '{parameter_name}'"
                    f" in {class_member}?",
                    f"Sure, the default value of '{parameter_name}' in {class_member}"
                    f" is {parameter_default}.",
                ),
                (
                    f"I need to know the default value of '{parameter_name}' in {class_member}.",
                    f"The default value of '{parameter_name}' in {class_member}"
                    f" is {parameter_default}.",
                ),
                (
                    f"Can you provide the default value of '{parameter_name}' in {class_member}?",
                    f"Yes, the default value of '{parameter_name}' in {class_member}"
                    f" is {parameter_default}.",
                ),
                (
                    f"Please, disclose the default value of '{parameter_name}' in {class_member}.",
                    f"Certainly, the default value of '{parameter_name}' in {class_member}"
                    f" is {parameter_default}.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_parameter_defaults_question_answer_pairs)
            )

        if (parameter_annotation := class_parameter.parameter_annotation) is inspect._empty:
            class_parameter_types_question_answer_pairs = [
                (
                    f"Name type hint for '{parameter_name}' in {class_member}.",
                    f"Parameter '{parameter_name}' does not have a type annotation.",
                ),
                (
                    f"What is the type hint for '{parameter_name}' in {class_member}?",
                    f"There is no type annotation for the parameter '{parameter_name}'.",
                ),
                (
                    f"Can you tell me the type hint for '{parameter_name}' in {class_member}?",
                    f"The parameter '{parameter_name}' is not annotated with a type.",
                ),
                (
                    f"I'm looking for the type hint for '{parameter_name}' in {class_member}."
                    " Can you help?",
                    f"Sure, the parameter '{parameter_name}' does not have a type annotation.",
                ),
                (
                    f"Could you provide the type hint for '{parameter_name}' in {class_member}?",
                    f"Unfortunately, parameter '{parameter_name}' does not have type annotation.",
                ),
                (
                    f"I need to know the type hint for '{parameter_name}' in {class_member}.",
                    f"The parameter '{parameter_name}' does not come with a type annotation.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_parameter_types_question_answer_pairs)
            )
        else:
            class_parameter_types_question_answer_pairs = [
                (
                    f"Name type hint for '{parameter_name}' in {class_member}.",
                    f"'parameter_name' parameter has '{parameter_annotation}' as type annotation.",
                ),
                (
                    f"What is the type hint for '{parameter_name}' in {class_member}?",
                    f"The type hint for 'parameter_name' in {class_member} is"
                    f" '{parameter_annotation}'.",
                ),
                (
                    f"Could you tell me the type hint for '{parameter_name}' in {class_member}?",
                    f"Sure, the type hint for 'parameter_name' in {class_member} is"
                    f" '{parameter_annotation}'.",
                ),
                (
                    f"I need to know the type hint for '{parameter_name}' in {class_member}.",
                    f"The type hint for 'parameter_name' in {class_member} is"
                    f" '{parameter_annotation}'.",
                ),
                (
                    f"Identify the type hint for '{parameter_name}' in {class_member}.",
                    f"The type hint for 'parameter_name' in {class_member} is"
                    f" '{parameter_annotation}'.",
                ),
                (
                    f"Can you specify the type hint for '{parameter_name}' in {class_member}?",
                    f"Yes, the type hint for 'parameter_name' in {class_member} is"
                    f" '{parameter_annotation}'.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_parameter_types_question_answer_pairs)
            )

        if not (parameter_summary := class_parameter.parameter_summary):
            class_parameter_summary_question_answer_pairs = [
                (
                    f"What does '{parameter_name}' do in {class_member}?",
                    f"Docstring of {class_member} does not describe '{parameter_name}'.",
                ),
                (
                    f"Can you explain the role of '{parameter_name}' in {class_member}?",
                    f"The docstring of {class_member} does not provide any information"
                    f" about '{parameter_name}'.",
                ),
                (
                    f"I'm trying to understand what '{parameter_name}' does in {class_member}."
                    " Can you help?",
                    f"Unfortunately, the docstring of {class_member} does not mention anything"
                    f" about '{parameter_name}'.",
                ),
                (
                    f"What is the function of '{parameter_name}' in {class_member}?",
                    f"There is no description of '{parameter_name}' in the docstring"
                    f" of {class_member}.",
                ),
                (
                    f"Could you tell me what '{parameter_name}' does in {class_member}?",
                    f"The docstring of {class_member} does not contain any details"
                    f" about '{parameter_name}'.",
                ),
                (
                    f"I'm curious about the purpose of '{parameter_name}' in {class_member}."
                    " Can you enlighten me?",
                    f"I'm sorry, but the docstring of {class_member} does not discuss"
                    f" '{parameter_name}'.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_parameter_summary_question_answer_pairs)
            )
        else:
            class_parameter_summary_question_answer_pairs = [
                (
                    f"What does '{parameter_name}' do in {class_member}?",
                    f"{class_member} documents role of '{parameter_name}'"
                    f" as '{parameter_summary}'.",
                ),
                (
                    f"Can you explain the role of '{parameter_name}' in {class_member}?",
                    f"Sure, {class_member} defines '{parameter_name}' as '{parameter_summary}'.",
                ),
                (
                    f"I'm curious about '{parameter_name}' in {class_member}. What does it do?",
                    f"In {class_member}, '{parameter_name}' is documented"
                    f" as '{parameter_summary}'.",
                ),
                (
                    f"Could you tell me what '{parameter_name}' does in {class_member}?",
                    f"Of course, in {class_member}, '{parameter_name}' is described"
                    f" as '{parameter_summary}'.",
                ),
                (
                    f"What's the function of '{parameter_name}' in {class_member}?",
                    f"{class_member} describes the function of '{parameter_name}'"
                    f" as '{parameter_summary}'.",
                ),
                (
                    f"I'd like to know the purpose of '{parameter_name}' in {class_member}.",
                    f"In {class_member}, the purpose of '{parameter_name}' is defined"
                    f" as '{parameter_summary}'.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_parameter_summary_question_answer_pairs)
            )

    if not (class_methods := member_type_details.class_methods):
        class_method_names_question_answer_pairs = [
            (
                f"List names of the public methods of {class_member}.",
                f"{class_member} does not have any public methods (not starting with '_').",
            ),
            (
                f"Can you provide the names of the public methods for {class_member}?",
                f"Unfortunately, {class_member} does not have any public methods"
                " (not starting with '_').",
            ),
            (
                f"What are the public methods of {class_member}?",
                f"There are no public methods (not starting with '_') in {class_member}.",
            ),
            (
                f"I need to know the public methods of {class_member}. Can you list them?",
                f"I'm sorry, but {class_member} does not have any public methods"
                " (not starting with '_').",
            ),
            (
                f"Could you list the public methods of {class_member}?",
                f"{class_member} does not contain any public methods (not starting with '_').",
            ),
            (
                f"Show me the public methods of {class_member}.",
                f"It appears that {class_member} does not have any public methods"
                " (not starting with '_').",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_method_names_question_answer_pairs))
    else:
        class_public_methods = " ".join(
            f"{counter + 1}. {class_method.method_name}"
            for counter, class_method in enumerate(class_methods)
        )
        class_method_names_question_answer_pairs = [
            (
                f"List names of the public methods of {class_member}.",
                f"Here are the public methods of {class_member} (not starting with '_'):"
                f" {class_public_methods}.",
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
        class_member_dataset.extend(generate_dataset(class_method_names_question_answer_pairs))

    for class_method in class_methods:
        method_name = class_method.method_name

        if not (method_parameters := class_method.method_parameters):
            class_method_parameters_question_answer_pairs = [
                (
                    f"What arguments do '{method_name}' method of {class_member} accept?",
                    f"'{method_name}' method does not take any parameters.",
                ),
                (
                    f"Can you tell me the parameters that '{method_name}' method"
                    f" of {class_member} requires?",
                    f"The '{method_name}' method does not require any parameters.",
                ),
                (
                    f"What are the inputs for the '{method_name}' method in {class_member}?",
                    f"There are no inputs for the '{method_name}' method in {class_member}.",
                ),
                (
                    f"Does the '{method_name}' method of {class_member} need any arguments?",
                    f"No, '{method_name}' method of {class_member} does not need any arguments.",
                ),
                (
                    f"What parameters should I pass to '{method_name}' method of {class_member}?",
                    f"You don't need to pass any parameters to the '{method_name}' method"
                    f" of {class_member}.",
                ),
                (
                    f"What are required arguments for '{method_name}' method of {class_member}?",
                    f"'{method_name}' method of {class_member} does not require any arguments.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_method_parameters_question_answer_pairs)
            )
        else:
            class_method_parameters = " ".join(
                f"{counter + 1}. {method_parameter}"
                for counter, method_parameter in enumerate(method_parameters)
            )
            class_method_parameters_question_answer_pairs = [
                (
                    f"What arguments do '{method_name}' method of {class_member} accept?",
                    f"'{method_name}' of {class_member} takes the following parameters:"
                    f" {class_method_parameters}.",
                ),
                (
                    f"Can you tell me the parameters that '{method_name}' method"
                    f" of {class_member} requires?",
                    f"Sure, the method '{method_name}' of {class_member} requires"
                    f" these parameters: {class_method_parameters}.",
                ),
                (
                    f"I need to know the arguments for '{method_name}' method in {class_member}.",
                    f"The '{method_name}' method in {class_member} has these arguments:"
                    f" {class_method_parameters}.",
                ),
                (
                    f"What are the parameters for '{method_name}' in {class_member}?",
                    f"The parameters for '{method_name}' in {class_member} are:"
                    f" {class_method_parameters}.",
                ),
                (
                    f"Could you list the arguments that the '{method_name}' method"
                    f" of {class_member} takes?",
                    f"Certainly, the '{method_name}' method of {class_member} takes"
                    f" these arguments: {class_method_parameters}.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_method_parameters_question_answer_pairs)
            )

        if not (method_summary := class_method.method_summary):
            class_method_summary_question_answer_pairs = [
                (
                    f"What does '{method_name}' method do in {class_member}?",
                    f"Docstring of '{method_name}' method is missing.",
                ),
                (
                    f"Can you explain functionality of '{method_name}' method in {class_member}?",
                    f"The docstring for '{method_name}' method is not available.",
                ),
                (
                    f"I'm trying to understand what '{method_name}' method does in {class_member}."
                    " Can you help?",
                    f"Unfortunately, the docstring for '{method_name}' method is not provided.",
                ),
                (
                    f"Could you describe the role of '{method_name}' method in {class_member}?",
                    f"There is no docstring available for '{method_name}' method.",
                ),
                (
                    f"I'm not sure what '{method_name}' method in {class_member} does."
                    " Can you clarify?",
                    f"The '{method_name}' method lacks a docstring.",
                ),
                (
                    f"What's the purpose of '{method_name}' method in {class_member}?",
                    f"The '{method_name}' method doesn't have a docstring.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_method_summary_question_answer_pairs)
            )
        else:
            class_method_summary_question_answer_pairs = [
                (
                    f"What does '{method_name}' method do in {class_member}?",
                    f"Based on method docstring, its role is to '{method_summary}'.",
                ),
                (
                    f"Can you explain the function of '{method_name}' method in {class_member}?",
                    f"Sure, according to method docstring, it is designed to '{method_summary}'.",
                ),
                (
                    f"I'm curious about the '{method_name}' method in {class_member}."
                    " What's its purpose?",
                    f"Well, if we look at the method docstring, we can see that"
                    f" it's meant to '{method_summary}'.",
                ),
                (
                    f"Could you tell me what the '{method_name}' method in {class_member} does?",
                    f"Of course, the method docstring indicates that its function"
                    f" is to '{method_summary}'.",
                ),
                (
                    f"I'd like to understand role of '{method_name}' method in {class_member}.",
                    f"Certainly, method docstring reveals that its job is to '{method_summary}'.",
                ),
                (
                    f"What's the functionality of the '{method_name}' method in {class_member}?",
                    f"As per the method docstring, it's designed to '{method_summary}'.",
                ),
            ]
            class_member_dataset.extend(
                generate_dataset(class_method_summary_question_answer_pairs)
            )

    if not (class_attributes := member_type_details.class_attributes):
        class_attribute_names_question_answer_pairs = [
            (
                f"Are there any public attributes of {class_member}?",
                f"{class_member} has no public attributes (not starting with '_').",
            ),
            (
                f"Does {class_member} have any public attributes?",
                f"No, {class_member} does not have any public attributes (not starting with '_').",
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
                f"It's not possible to find any public attributes (not starting with '_')"
                f" in {class_member}.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_attribute_names_question_answer_pairs))
    else:
        class_public_attributes = " ".join(
            f"{counter + 1}. {class_attribute.attribute_name}"
            for counter, class_attribute in enumerate(class_attributes)
        )
        class_attribute_names_question_answer_pairs = [
            (
                f"Are there any public attributes of {class_member}?",
                f"These are the public (not starting with '_') attributes of {class_member}:"
                f" {class_public_attributes}.",
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
                f"Sure, the public attributes of {class_member} (not starting with '_') are:"
                f" {class_public_attributes}.",
            ),
            (
                f"Could you tell me the public attributes of {class_member}?",
                f"Of course, public attributes of {class_member} (not starting with '_') are:"
                f" {class_public_attributes}.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_attribute_names_question_answer_pairs))

    if not (class_summary := member_type_details.class_summary):
        class_summary_question_answer_pairs = [
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
                f"Docstring of {class_member} doesn't have a short explanation of its objective.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_summary_question_answer_pairs))
    else:
        class_summary_question_answer_pairs = [
            (
                f"What does {class_member} do in short?",
                f"Based on documentation, objective of {class_member} is to: '{class_summary}'.",
            ),
            (
                f"Can you briefly explain the function of {class_member}?",
                f"Sure, according to the documentation, {class_member} is"
                f" designed to: '{class_summary}'.",
            ),
            (
                f"I'm curious about {class_member}, what's its purpose?",
                f"Well, as per the documentation, {class_member} aims to: '{class_summary}'.",
            ),
            (
                f"Could you give me a quick rundown on what {class_member} does?",
                f"Absolutely, the documentation states that the role of {class_member} is"
                f" to: '{class_summary}'.",
            ),
            (
                f"What's the role of {class_member} in a nutshell?",
                f"The documentation indicates that the purpose of {class_member} is"
                f" to: '{class_summary}'.",
            ),
            (
                f"Can you summarize the function of {class_member}?",
                f"Of course, the documentation outlines that {class_member} is intended"
                f" to: '{class_summary}'.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_summary_question_answer_pairs))

    if not (class_notes := member_type_details.class_notes):
        class_notes_question_answer_pairs = [
            (
                f"Mention any specific details for {class_member} to be aware of.",
                f"Docstring of {class_member} does not note on specific details.",
            ),
            (
                f"What are the specific details to be aware of for {class_member}?",
                f"There are no specific details noted in the docstring of {class_member}.",
            ),
            (
                f"Could you tell me the specific details for {class_member} that"
                " I should be aware of?",
                f"The docstring of {class_member} doesn't provide any specific details"
                " to be aware of.",
            ),
            (
                f"Are there any specific details for {class_member} that I need to know?",
                f"No specific details are mentioned in the docstring of {class_member}.",
            ),
            (
                f"I need to know the specific details for {class_member}. Can you provide them?",
                f"Unfortunately, the docstring of {class_member} does not contain"
                " any specific details.",
            ),
            (
                f"Can you specify any details for {class_member} that I should be aware of?",
                f"The docstring of {class_member} does not specify any details to be aware of.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_notes_question_answer_pairs))
    else:
        class_notes_question_answer_pairs = [
            (
                f"Mention any specific details for {class_member} to be aware of.",
                f"The {class_member} docstring highlights the following: '{class_notes}'.",
            ),
            (
                f"What are the specific details that {class_member} should be aware of?",
                f"The details that {class_member} should be aware of are highlighted"
                f" in the docstring: '{class_notes}'.",
            ),
            (
                f"Could you specify the details for {class_member} to take note of?",
                f"Sure, the docstring for {class_member} specifies"
                f" the following details: '{class_notes}'.",
            ),
            (
                f"Can you list the details for {class_member} to keep in mind?",
                f"Certainly, the docstring for {class_member} lists"
                f" the following details: '{class_notes}'.",
            ),
            (
                f"What should {class_member} be mindful of?",
                f"The docstring for {class_member} mentions"
                f" the following points to be mindful of: '{class_notes}'.",
            ),
            (
                f"What details does the user of {class_member} need to know?",
                f"User of {class_member} needs to know the following details"
                f" as mentioned in the docstring: '{class_notes}'.",
            ),
        ]
        class_member_dataset.extend(generate_dataset(class_notes_question_answer_pairs))

    return class_member_dataset


@pydantic.validate_call(validate_return=True)
def generate_function_member_dataset(  # noqa: C901, PLR0912, PLR0915
    function_member: str, member_type_details: FunctionDetails
) -> list[Document]:
    function_member_dataset: list[Document] = []

    if not (function_parameters := member_type_details.function_parameters):
        function_parameters_question_answer_pairs = [
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
        function_member_dataset.extend(generate_dataset(function_parameters_question_answer_pairs))
    else:
        function_parameter_names = " ".join(
            f"{counter + 1}. {function_parameter.parameter_details}"
            for counter, function_parameter in enumerate(function_parameters)
        )
        function_parameters_question_answer_pairs = [
            (
                f"List various parameters of {function_member}.",
                f"Different parameters of {function_member} are"
                f" as follows: {function_parameter_names}.",
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
                f"The parameters of {function_member} are as follows: {function_parameter_names}.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_parameters_question_answer_pairs))

    for function_parameter in function_parameters:
        parameter_name = function_parameter.parameter_name

        if (parameter_default := function_parameter.parameter_default) is inspect._empty:
            function_parameter_defaults_question_answer_pairs = [
                (
                    f"Default value of '{parameter_name}' in {function_member}?",
                    f"'{parameter_name}' argument does not have a default value.",
                ),
                (
                    f"What is the default value for '{parameter_name}' in {function_member}?",
                    f"The argument '{parameter_name}' does not come with a default value.",
                ),
                (
                    f"Could you tell me default value of '{parameter_name}' in {function_member}?",
                    f"Sure, the '{parameter_name}' argument does not possess a default value.",
                ),
                (
                    f"I'm curious about default value of '{parameter_name}' in {function_member}.",
                    f"In response to your curiosity, '{parameter_name}' argument"
                    " is not assigned a default value.",
                ),
                (
                    f"I'd like to know the default value"
                    f" of '{parameter_name}' in {function_member}.",
                    f"To answer your query, '{parameter_name}' argument"
                    " does not hold a default value.",
                ),
                (
                    f"Can you inform me about the default value"
                    f" of '{parameter_name}' in {function_member}?",
                    f"Certainly, '{parameter_name}' argument does not contain a default value.",
                ),
            ]
            function_member_dataset.extend(
                generate_dataset(function_parameter_defaults_question_answer_pairs)
            )
        else:
            function_parameter_defaults_question_answer_pairs = [
                (
                    f"Default value of '{parameter_name}' in {function_member}?",
                    f"'{parameter_name}' parameter has default value of {parameter_default}.",
                ),
                (
                    f"What is the default value for '{parameter_name}' in {function_member}?",
                    f"The default value for '{parameter_name}' in {function_member}"
                    f" is {parameter_default}.",
                ),
                (
                    f"Could you tell me default value of '{parameter_name}' in {function_member}?",
                    f"Sure, the default value of '{parameter_name}' in {function_member}"
                    f" is {parameter_default}.",
                ),
                (
                    f"I would like to know the default value"
                    f" of '{parameter_name}' in {function_member}.",
                    f"The '{parameter_name}' parameter in {function_member}"
                    f" has a default value of {parameter_default}.",
                ),
                (
                    f"Can you inform me about the default value"
                    f" of '{parameter_name}' in {function_member}?",
                    f"Of course, the '{parameter_name}' parameter in {function_member}"
                    f" defaults to {parameter_default}.",
                ),
                (
                    f"I'm interested in default value of '{parameter_name}' in {function_member}.",
                    f"The default value of the '{parameter_name}' parameter"
                    f" in {function_member} is {parameter_default}.",
                ),
            ]
            function_member_dataset.extend(
                generate_dataset(function_parameter_defaults_question_answer_pairs)
            )

        if (parameter_annotation := function_parameter.parameter_annotation) is inspect._empty:
            function_parameter_types_question_answer_pairs = [
                (
                    f"What is type annotation of '{parameter_name}' in {function_member}?",
                    f"'{parameter_name}' parameter does not have a type annotation.",
                ),
                (
                    f"Can you tell me type annotation of '{parameter_name}' in {function_member}?",
                    f"The parameter '{parameter_name}' does not have a type annotation.",
                ),
                (
                    f"I'm curious about the type annotation of '{parameter_name}'"
                    f" in {function_member}. Can you provide some information?",
                    f"Sure, the parameter '{parameter_name}' does not have a type annotation.",
                ),
                (
                    f"Do you have any information on the type annotation of '{parameter_name}'"
                    f" in {function_member}?",
                    f"Yes, the parameter '{parameter_name}' does not have a type annotation.",
                ),
                (
                    f"Could you inform me about the type annotation of '{parameter_name}'"
                    f" in {function_member}?",
                    f"Certainly, parameter '{parameter_name}' does not have a type annotation.",
                ),
                (
                    f"I'd like to know the type annotation of '{parameter_name}'"
                    f" in {function_member}.",
                    f"The parameter '{parameter_name}' you're asking about"
                    " does not have a type annotation.",
                ),
            ]
            function_member_dataset.extend(
                generate_dataset(function_parameter_types_question_answer_pairs)
            )
        else:
            function_parameter_types_question_answer_pairs = [
                (
                    f"What is type annotation of '{parameter_name}' in {function_member}?",
                    f"Type annotation of '{parameter_name}' argument is '{parameter_annotation}'.",
                ),
                (
                    f"Can you tell me type annotation of '{parameter_name}' in {function_member}?",
                    f"Sure, the type annotation of '{parameter_name}' argument is"
                    f" '{parameter_annotation}'.",
                ),
                (
                    f"I'm curious about the type annotation of '{parameter_name}'"
                    f" in {function_member}. What is it?",
                    f"The type annotation of '{parameter_name}' argument in {function_member}"
                    f" is '{parameter_annotation}'.",
                ),
                (
                    f"Do you know the type annotation of '{parameter_name}' in {function_member}?",
                    f"Yes, the type annotation of '{parameter_name}' argument"
                    f" is '{parameter_annotation}'.",
                ),
                (
                    f"Could you inform me about the type annotation of '{parameter_name}'"
                    f" in {function_member}?",
                    f"Of course, the type annotation of '{parameter_name}' argument"
                    f" is '{parameter_annotation}'.",
                ),
                (
                    f"What's the type annotation for '{parameter_name}' in the function"
                    f" {function_member}?",
                    f"The type annotation for '{parameter_name}' in the function"
                    f" {function_member} is '{parameter_annotation}'.",
                ),
            ]
            function_member_dataset.extend(
                generate_dataset(function_parameter_types_question_answer_pairs)
            )

        if not (parameter_summary := function_parameter.parameter_summary):
            function_parameter_summary_question_answer_pairs = [
                (
                    f"What is '{parameter_name}' parameter for in {function_member}?",
                    f"Docstring of {function_member} lacks a description.",
                ),
                (
                    f"Can you explain the purpose of '{parameter_name}' in {function_member}?",
                    f"The docstring of {function_member} doesn't provide a description.",
                ),
                (
                    f"I'm not sure what '{parameter_name}' does in {function_member}."
                    " Can you help?",
                    f"Unfortunately, the docstring of {function_member} doesn't"
                    " include a description.",
                ),
                (
                    f"Could you clarify the role of '{parameter_name}' in {function_member}?",
                    f"The description is missing in the docstring of {function_member}.",
                ),
                (
                    f"I'm confused about the '{parameter_name}' parameter in {function_member}."
                    " What does it do?",
                    f"The docstring of {function_member} doesn't contain a description.",
                ),
                (
                    f"What does '{parameter_name}' parameter do in {function_member}?",
                    f"There's no description in the docstring of {function_member}.",
                ),
            ]
            function_member_dataset.extend(
                generate_dataset(function_parameter_summary_question_answer_pairs)
            )
        else:
            function_parameter_summary_question_answer_pairs = [
                (
                    f"What is '{parameter_name}' parameter for in {function_member}?",
                    f"Based on {function_member} docstring, its role is '{parameter_summary}'.",
                ),
                (
                    f"Can you explain the role of '{parameter_name}' in {function_member}?",
                    f"Sure, according to the docstring of {function_member},"
                    f" '{parameter_name}' is used for '{parameter_summary}'.",
                ),
                (
                    f"I'm curious about the '{parameter_name}' parameter in {function_member}."
                    " What does it do?",
                    f"Well, if you look at the docstring of {function_member}, you'll see"
                    f" that '{parameter_name}' is responsible for '{parameter_summary}'.",
                ),
                (
                    f"Could you tell me the purpose of '{parameter_name}' in {function_member}?",
                    f"Of course, the docstring of {function_member} indicates"
                    f" that '{parameter_name}' serves the purpose of '{parameter_summary}'.",
                ),
                (
                    f"What's the function of '{parameter_name}' parameter in {function_member}?",
                    f"As per the docstring of {function_member},"
                    f" '{parameter_name}' functions as '{parameter_summary}'.",
                ),
                (
                    f"I'd like to know what '{parameter_name}' does in {function_member}.",
                    f"Sure thing, the docstring of {function_member} states"
                    f" that '{parameter_name}' does '{parameter_summary}'.",
                ),
            ]
            function_member_dataset.extend(
                generate_dataset(function_parameter_summary_question_answer_pairs)
            )

    if (
        returns_annotation := member_type_details.function_returns.returns_annotation
    ) is inspect._empty:
        function_return_type_question_answer_pairs = [
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
                f"I'm curious about the return type annotation of {function_member}. What is it?",
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
        function_member_dataset.extend(
            generate_dataset(function_return_type_question_answer_pairs)
        )
    else:
        function_return_type_question_answer_pairs = [
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
                "Of course, the return type annotation"
                f" for {function_member} is '{returns_annotation}'.",
            ),
            (
                f"I'm curious about the return type annotation of {function_member}.",
                f"The return type annotation for {function_member} is '{returns_annotation}'.",
            ),
        ]
        function_member_dataset.extend(
            generate_dataset(function_return_type_question_answer_pairs)
        )

    if not (returns_summary := member_type_details.function_returns.returns_summary):
        function_return_summary_question_answer_pairs = [
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
        function_member_dataset.extend(
            generate_dataset(function_return_summary_question_answer_pairs)
        )
    else:
        function_return_summary_question_answer_pairs = [
            (
                f"What does {function_member} return?",
                f"Based on {function_member} docstring, the return contains: '{returns_summary}'.",
            ),
            (
                f"Can you tell me what {function_member} returns?",
                f"Sure, according to the docstring of {function_member},"
                f" it returns: '{returns_summary}'.",
            ),
            (
                f"I'm curious about what {function_member} returns. Can you help?",
                f"Absolutely! The docstring of {function_member} indicates that"
                f" it returns: '{returns_summary}'.",
            ),
            (
                f"Do you know what {function_member} returns?",
                f"Yes, the docstring of {function_member} states that"
                f" it returns: '{returns_summary}'.",
            ),
            (
                f"I'd like to know what {function_member} returns.",
                f"Of course, the docstring of {function_member} reveals that"
                f" its return contains: '{returns_summary}'.",
            ),
            (
                f"Could you inform me about the return of {function_member}?",
                f"Certainly, the docstring of {function_member} specifies that"
                f" it returns: '{returns_summary}'.",
            ),
        ]
        function_member_dataset.extend(
            generate_dataset(function_return_summary_question_answer_pairs)
        )

    if not (function_summary := member_type_details.function_summary):
        function_summary_question_answer_pairs = [
            (
                f"Summarise role of {function_member} in short.",
                f"{function_member} docstring lacks a summary of its objective.",
            ),
            (
                f"Can you briefly explain the role of {function_member}?",
                f"The docstring of {function_member} doesn't provide"
                " a brief explanation of its purpose.",
            ),
            (
                f"What is the purpose of {function_member} as per its docstring?",
                f"The docstring of {function_member} doesn't clearly state its purpose.",
            ),
            (
                f"Could you provide a summary of {function_member}'s objective?",
                f"The objective of {function_member} is not summarised in its docstring.",
            ),
            (
                f"What does {function_member} do according to its docstring?",
                f"According to its docstring, {function_member}'s role is not summarised.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_summary_question_answer_pairs))
    else:
        function_summary_question_answer_pairs = [
            (
                f"Summarise role of {function_member} in short.",
                f"Based on docstring, objective of {function_member} is to: '{function_summary}'.",
            ),
            (
                f"Can you briefly explain the role of {function_member}?",
                "Sure, according to the docstring,"
                f" the purpose of {function_member} is: '{function_summary}'.",
            ),
            (
                f"What does {function_member} do, in a nutshell?",
                f"In a nutshell, {function_member} is designed to: '{function_summary}',"
                " as per the docstring.",
            ),
            (
                f"Could you provide a short summary of {function_member}'s role?",
                f"Certainly, {function_member} aims to: '{function_summary}',"
                " as described in the docstring.",
            ),
            (
                f"I need a brief explanation of what {function_member} does.",
                f"Of course, {function_member} is intended to: '{function_summary}',"
                " as stated in the docstring.",
            ),
            (
                f"In brief, what is the role of {function_member}?",
                f"Briefly, the role of {function_member} is to: '{function_summary}',"
                " according to the docstring.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_summary_question_answer_pairs))

    if not (function_raises := member_type_details.function_raises):
        function_raise_types_question_answer_pairs = [
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
                f"According to the docstring, {function_member} does not"
                " raise any specific exceptions.",
            ),
            (
                f"I want to know if {function_member} raises any specific exceptions."
                " Can you confirm?",
                f"I can confirm that the docstring of {function_member} does not mention"
                " any specific exceptions.",
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
        function_member_dataset.extend(
            generate_dataset(function_raise_types_question_answer_pairs)
        )
    else:
        function_raise_types = " ".join(
            f"{counter + 1}. {function_raise.raises_details}"
            for counter, function_raise in enumerate(function_raises)
        )
        function_raise_types_question_answer_pairs = [
            (
                f"Does {function_member} raise any specific exception?",
                f"Based on {function_member}'s docstring,"
                f" it can raise the following: {function_raise_types}.",
            ),
            (
                f"Can you tell me if {function_member} raises any specific exceptions?",
                f"Yes, according to the docstring of {function_member},"
                f" it can raise these exceptions: {function_raise_types}.",
            ),
            (
                f"What exceptions, if any, does {function_member} raise?",
                f"{function_member} can raise these exceptions"
                f" as per its docstring: {function_raise_types}.",
            ),
            (
                f"I need to know if {function_member} throws any specific exceptions."
                " Can you help?",
                f"Sure, {function_member} can throw the following exceptions"
                f" according to its docstring: {function_raise_types}.",
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
        function_member_dataset.extend(
            generate_dataset(function_raise_types_question_answer_pairs)
        )

    if not (function_warns := member_type_details.function_warns):
        function_warn_types_question_answer_pairs = [
            (
                f"Does {function_member} throw any specific warnings?",
                f"{function_member}'s docstring lacks any mention of specific warnings.",
            ),
            (
                f"Are there any specific warnings that {function_member} throws?",
                f"There are no specific warnings mentioned in the docstring of {function_member}.",
            ),
            (
                f"Can you tell me if {function_member} throws any specific warnings?",
                f"According to the docstring of {function_member},"
                " it doesn't throw any specific warnings.",
            ),
            (
                f"I want to know if {function_member} throws any specific warnings. Can you help?",
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
        function_member_dataset.extend(generate_dataset(function_warn_types_question_answer_pairs))
    else:
        function_warn_types = " ".join(
            f"{counter + 1}. {function_warn.warns_details}"
            for counter, function_warn in enumerate(function_warns)
        )
        function_warn_types_question_answer_pairs = [
            (
                f"Does {function_member} throw any specific warnings?",
                f"Based on the docstring, {function_member} can throw"
                f" the following warnings: {function_warn_types}.",
            ),
            (
                f"Can you tell me if {function_member} throws any specific warnings?",
                f"Yes, according to the docstring, {function_member} may throw"
                f" these specific warnings: {function_warn_types}.",
            ),
            (
                f"I'm curious, does {function_member} generate any particular warnings?",
                f"Indeed, the docstring indicates that {function_member} can generate"
                f" these specific warnings: {function_warn_types}.",
            ),
            (
                f"What specific warnings, if any, does {function_member} throw?",
                f"{function_member} throws the following specific warnings"
                f" as per the docstring: {function_warn_types}.",
            ),
            (
                f"Could {function_member} possibly throw any specific warnings?",
                f"Yes, it could. The docstring of {function_member} mentions"
                f" these specific warnings: {function_warn_types}.",
            ),
            (
                f"Are there any specific warnings that {function_member} throws?",
                f"Yes, there are. The docstring for {function_member} lists"
                f" the following specific warnings: {function_warn_types}.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_warn_types_question_answer_pairs))

    if not (function_notes := member_type_details.function_notes):
        function_notes_question_answer_pairs = [
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
                f"The docstring of {function_member} does not contain"
                " any specific details to be aware of.",
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
        function_member_dataset.extend(generate_dataset(function_notes_question_answer_pairs))
    else:
        function_notes_question_answer_pairs = [
            (
                f"Is there any specific details for {function_member} to be aware of?",
                f"The {function_member}'s docstring highlights the following: '{function_notes}'.",
            ),
            (
                f"What should I know about {function_member}?",
                f"You should be aware that the docstring of {function_member} includes"
                f" the following details: '{function_notes}'.",
            ),
            (
                f"Could you provide some details about {function_member}?",
                f"Sure, the docstring of {function_member} provides"
                f" the following information: '{function_notes}'.",
            ),
            (
                f"What are the important details of {function_member}?",
                f"The important details of {function_member} are highlighted"
                f" in its docstring: '{function_notes}'.",
            ),
            (
                f"Can you tell me more about {function_member}?",
                f"Of course, the docstring of {function_member} contains"
                f" the following details: '{function_notes}'.",
            ),
            (
                f"I need information about {function_member}.",
                f"The docstring of {function_member} contains"
                f" the following information: '{function_notes}'.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_notes_question_answer_pairs))

    if not (function_references := member_type_details.function_references):
        function_references_question_answer_pairs = [
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
                f"Unfortunately, the documentation for {function_member}"
                " does not contain any references.",
            ),
            (
                "Could you tell me if there are any references"
                f" in the {function_member} documentation?",
                f"I'm sorry, but the documentation for {function_member}"
                " does not contain any references.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_references_question_answer_pairs))
    else:
        function_references_question_answer_pairs = [
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
        function_member_dataset.extend(generate_dataset(function_references_question_answer_pairs))

    if not (function_examples := member_type_details.function_examples):
        function_examples_question_answer_pairs = [
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
                f"I'm looking for an example of {function_member} in the docstring, is there one?",
                f"I'm sorry, but docstring for {function_member} does not provide any examples.",
            ),
            (
                f"Are there any examples provided in the docstring for {function_member}?",
                f"No examples are provided in the docstring for {function_member}.",
            ),
            (
                f"Could you tell me if there's an example for {function_member} in the docstring?",
                f"I regret to inform you that the docstring for {function_member}"
                " does not have any examples.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_examples_question_answer_pairs))
    else:
        function_examples_question_answer_pairs = [
            (
                f"Is there any example for {function_member}?",
                f"Documentation of {function_member} contains"
                f" these examples: '{function_examples}'.",
            ),
            (
                f"Can you provide an example of {function_member}?",
                f"Sure, you can find examples of {function_member} in"
                f" its documentation: '{function_examples}'.",
            ),
            (
                f"I'm looking for examples of {function_member}, can you help?",
                f"Absolutely, the examples for {function_member} are available in"
                f" its documentation: '{function_examples}'.",
            ),
            (
                f"Where can I find examples for {function_member}?",
                f"You can find examples for {function_member} in"
                f" its documentation: '{function_examples}'.",
            ),
            (
                f"Could you show me some examples of {function_member}?",
                f"Of course, the documentation of {function_member} includes"
                f" these examples: '{function_examples}'.",
            ),
            (
                f"I need examples for {function_member}, where can I find them?",
                f"You can find examples for {function_member} in"
                f" its documentation: '{function_examples}'.",
            ),
        ]
        function_member_dataset.extend(generate_dataset(function_examples_question_answer_pairs))

    return function_member_dataset


@pydantic.validate_call(validate_return=True)
def generate_member_dataset(member_details: MemberDetails) -> list[Document]:
    member_name = member_details.member_name
    member_full_name = member_details.member_qualified_name

    member_dataset: list[Document] = []

    module_parent_question_answer_pairs = [
        (
            f"What is the parent module of '{member_name}'?",
            f"'{member_details.member_module}' is the name of its parent module.",
        ),
        (
            f"Can you tell me the parent module of '{member_name}'?",
            f"Sure, the parent module of '{member_name}' is '{member_details.member_module}'.",
        ),
        (
            f"I'm trying to find the parent module of '{member_name}', can you help?",
            f"Of course, parent module of '{member_name}' is '{member_details.member_module}'.",
        ),
        (
            f"Do you know the parent module of '{member_name}'?",
            f"Yes, the parent module of '{member_name}' is '{member_details.member_module}'.",
        ),
        (
            f"I need to know the parent module of '{member_name}', can you provide that?",
            f"Absolutely, parent module of '{member_name}' is '{member_details.member_module}'.",
        ),
        (
            f"Could you inform me about the parent module of '{member_name}'?",
            f"Certainly, '{member_details.member_module}' is parent module of '{member_name}'.",
        ),
    ]
    member_dataset.extend(generate_dataset(module_parent_question_answer_pairs))

    member_full_name_question_answer_pairs = [
        (
            f"What is the full name of '{member_name}' member?",
            f"'{member_full_name}' is its fully qualified name.",
        ),
        (
            f"Can you tell me the full name of the member '{member_name}'?",
            f"Sure, the fully qualified name of the member is '{member_full_name}'.",
        ),
        (
            f"I need to know the full name of '{member_name}'. Can you help?",
            f"Of course, the full name of '{member_name}' is '{member_full_name}'.",
        ),
        (
            f"What's the fully qualified name for the member '{member_name}'?",
            f"The fully qualified name for '{member_name}' is '{member_full_name}'.",
        ),
        (
            f"Could you provide the full name of the member '{member_name}'?",
            f"Certainly, the full name of the member '{member_name}' is '{member_full_name}'.",
        ),
        (
            f"I'm looking for the full name of '{member_name}'. What is it?",
            f"The full name of '{member_name}' is '{member_full_name}'.",
        ),
    ]
    member_dataset.extend(generate_dataset(member_full_name_question_answer_pairs))

    member_hierarchy = " ".join(
        f"{counter + 1}. {node}" for counter, node in enumerate(member_details.member_hierarchy)
    )
    member_hierarchy_question_answer_pairs = [
        (
            f"What is the hierarchy of {member_name} member?",
            f"The hierarchy of '{member_name}' member is as follows: {member_hierarchy}.",
        ),
        (
            f"Can you explain the hierarchy of the {member_name} member?",
            f"Sure, the hierarchy of the '{member_name}' member is: {member_hierarchy}.",
        ),
        (
            f"Could you tell me the hierarchy of {member_name} member?",
            f"Of course, the hierarchy of '{member_name}' member is: {member_hierarchy}.",
        ),
        (
            f"I would like to know the hierarchy of {member_name} member. Can you provide that?",
            f"Absolutely, the hierarchy of '{member_name}' member is: {member_hierarchy}.",
        ),
        (
            f"Please provide the hierarchy of {member_name} member.",
            f"The hierarchy of '{member_name}' member is: {member_hierarchy}.",
        ),
        (
            f"I'm interested in the hierarchy of {member_name} member. Could you share it?",
            f"Sure, the hierarchy of '{member_name}' member is: {member_hierarchy}.",
        ),
    ]
    member_dataset.extend(generate_dataset(member_hierarchy_question_answer_pairs))

    if not (member_docstring := member_details.member_docstring):
        member_documentation_question_answer_pairs = [
            (
                f"What is the documentation of '{member_full_name}' member?",
                f"'{member_full_name}' member does not have any documentation.",
            ),
            (
                f"Can you provide the documentation for the member '{member_full_name}'?",
                f"Sorry, the member '{member_full_name}' does not have any documentation.",
            ),
            (
                f"Is there any documentation available for the '{member_full_name}' member?",
                f"No, there is no documentation available for the '{member_full_name}' member.",
            ),
            (
                f"Could you show me the documentation of the '{member_full_name}' member?",
                f"Unfortunately, the '{member_full_name}' member does not have any documentation.",
            ),
            (
                f"I'm looking for the documentation of '{member_full_name}' member. Can you help?",
                f"I'm sorry, but the '{member_full_name}' member does not have any documentation.",
            ),
        ]
        member_dataset.extend(generate_dataset(member_documentation_question_answer_pairs))
    else:
        member_documentation_question_answer_pairs = [
            (
                f"What does '{member_full_name}' member do?",
                f"Its documentation is as follows: '{member_docstring}'.",
            ),
            (
                f"Can you explain the function of the '{member_full_name}' member?",
                f"Sure, here is its documentation: '{member_docstring}'.",
            ),
            (
                f"I'm not sure what '{member_full_name}' member does. Can you clarify?",
                f"Of course, here's its documentation for clarification: '{member_docstring}'.",
            ),
            (
                f"Could you tell me about the '{member_full_name}' member?",
                f"Certainly, its documentation is: '{member_docstring}'.",
            ),
            (
                f"I need information on the '{member_full_name}' member.",
                f"Here's the documentation you need: '{member_docstring}'.",
            ),
            (
                f"What's the purpose of the '{member_full_name}' member?",
                f"The purpose is described in its documentation: '{member_docstring}'.",
            ),
        ]
        member_dataset.extend(generate_dataset(member_documentation_question_answer_pairs))

    if (member_type_details := member_details.member_type_details) is None:
        return member_dataset

    member_type = member_type_details.member_type

    member_type_question_answer_pairs = [
        (
            f"What is the type of '{member_name}' member?",
            f"'{member_full_name}' member is of '{member_type.value}' type.",
        ),
        (
            f"Can you tell me the type of the '{member_name}' member?",
            f"Sure, the '{member_full_name}' member is of '{member_type.value}' type.",
        ),
        (
            f"I would like to know the type of '{member_name}' member. Can you help?",
            f"Absolutely, the '{member_full_name}' member is of '{member_type.value}' type.",
        ),
        (
            f"Do you know the type of '{member_name}' member?",
            f"Yes, the '{member_full_name}' member is of '{member_type.value}' type.",
        ),
        (
            f"Could you inform me about the type of '{member_name}' member?",
            f"Of course, the '{member_full_name}' member is of '{member_type.value}' type.",
        ),
        (
            f"I'm curious about type of '{member_name}' member. Can you provide some information?",
            f"Certainly, the '{member_full_name}' member is of '{member_type.value}' type.",
        ),
    ]
    member_dataset.extend(generate_dataset(member_type_question_answer_pairs))

    match member_type:
        case MemberType.ENUM:
            member_dataset.extend(
                generate_enum_member_dataset(f"'{member_full_name}' enum", member_type_details)
            )
        case MemberType.CLASS:
            member_dataset.extend(
                generate_class_member_dataset(f"'{member_full_name}' class", member_type_details)
            )
        case MemberType.FUNCTION:
            member_dataset.extend(
                generate_function_member_dataset(
                    f"'{member_full_name}' function", member_type_details
                )
            )
        case _:
            raise ValueError("Unexpected member type: supports 'enum', 'class', 'function'")

    return member_dataset


__all__ = ["generate_member_dataset", "generate_module_dataset", "generate_package_dataset"]
