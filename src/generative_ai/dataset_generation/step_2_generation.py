import inspect

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


@pydantic.validate_call(validate_return=True)
def generate_dataset(questions: list[str], answers: list[str]) -> list[Document]:
    dataset = [
        Document(question=question, answer=answer)
        for question, answer in zip(questions, answers, strict=True)
    ]

    return dataset


@pydantic.validate_call(validate_return=True)
def generate_package_dataset(package_contents: Package) -> list[Document]:
    package_name = package_contents.package_name
    package_full_name = package_contents.package_qualified_name

    package_questions: list[str] = []
    package_answers: list[str] = []

    if (parent_package := package_contents.parent_package_name) is None:
        package_questions.append("What is the root package?")
        package_answers.append(f"'{package_name}' is the root package.")

        package_questions.append(f"Name parent package of '{package_name}'.")
        package_answers.append(f"Being the root package, '{package_name}' has no parent package.")
    else:
        package_questions.append(f"Name parent package of '{package_name}' sub-package.")
        package_answers.append(f"'{parent_package}' is the full name of its parent package.")

        package_questions.append(f"Tell the full name of '{package_name}' sub-package.")
        package_answers.append(
            f"'{package_full_name}' is the fully qualified name of '{package_name}'."
        )

        package_questions.append(f"What is the hierarchy of {package_name} package?")
        package_answers.append(
            " ".join(
                [
                    f"The hierarchy of '{package_name}' package is as follows:",
                    " ".join(
                        f"{counter + 1}. {node}"
                        for counter, node in enumerate(package_contents.package_hierarchy)
                    ),
                ]
            )
        )

    if not (children_sub_packages := package_contents.children_sub_packages_names):
        package_questions.append(f"List the sub-packages of '{package_full_name}' package.")
        package_answers.append(
            f"'{package_full_name}' package does not have any further sub-packages."
        )
    else:
        package_questions.append(f"List the sub-packages of '{package_full_name}' package.")
        package_answers.append(
            " ".join(
                [
                    f"Sub-packages of '{package_full_name}' package are as follows:",
                    " ".join(
                        f"{counter + 1}. {sub_package}"
                        for counter, sub_package in enumerate(children_sub_packages)
                    ),
                ]
            )
        )

    if not (children_modules := package_contents.children_modules_names):
        package_questions.append(f"What are the modules of '{package_full_name}' package?")
        package_answers.append(
            f"'{package_full_name}' does not have any direct modules under itself."
        )
    else:
        package_questions.append(f"What are the modules of '{package_full_name}' package?")
        package_answers.append(
            " ".join(
                [
                    f"Direct modules under '{package_full_name}' are as follows:",
                    " ".join(
                        f"{counter + 1}. {sub_package}"
                        for counter, sub_package in enumerate(children_modules)
                    ),
                ]
            )
        )

    if not (package_summary := package_contents.package_summary):
        package_questions.append(f"What does '{package_full_name}' package do?")
        package_answers.append(f"'{package_full_name}' package does not have any documentation.")
    else:
        package_questions.append(f"What does '{package_full_name}' package do?")
        package_answers.append(f"Its documentation is as follows: '{package_summary}'.")

    if not (package_exports := package_contents.package_all_exports):
        package_questions.append(
            f"What are the public members of the '{package_full_name}' package?"
        )
        package_answers.append("It does not have any public member exported through '__all__'.")
    else:
        package_questions.append(
            f"What are the public members of the '{package_full_name}' package?"
        )
        package_answers.append(
            " ".join(
                [
                    "It publicly exports the following members using '__all__':",
                    " ".join(
                        f"{counter + 1}. {package_export}"
                        for counter, package_export in enumerate(package_exports)
                    ),
                ]
            )
        )

    return generate_dataset(package_questions, package_answers)


@pydantic.validate_call(validate_return=True)
def generate_module_dataset(module_members: Module) -> list[Document]:
    module_name = module_members.module_name
    module_full_name = module_members.module_qualified_name

    module_questions: list[str] = []
    module_answers: list[str] = []

    module_questions.append(f"Can you tell the the parent package of '{module_name}' module?")
    module_answers.append(
        f"'{module_members.package_name}' is the parent package of '{module_name}'."
    )

    module_questions.append(f"Specify the full name of '{module_name}' module?")
    module_answers.append(
        f"'{module_full_name}' is fully qualified name for '{module_name}' module."
    )

    module_questions.append(f"What is the hierarchy of {module_name} module?")
    module_answers.append(
        " ".join(
            [
                f"The hierarchy of '{module_name}' module is as follows:",
                " ".join(
                    f"{counter + 1}. {node}"
                    for counter, node in enumerate(module_members.module_hierarchy)
                ),
            ]
        )
    )

    module_questions.append(f"List the members of '{module_name}' module.")
    module_answers.append(
        " ".join(
            [
                f"Members of '{module_full_name}' are as follows:",
                " ".join(
                    f"{counter + 1}. {member.member_name}"
                    for counter, member in enumerate(module_members.module_members)
                ),
            ]
        )
    )

    if not (module_summary := module_members.module_summary):
        module_questions.append(f"What is the '{module_full_name}' module for?")
        module_answers.append(f"'{module_name}' member does not have any documentation.")
    else:
        module_questions.append(f"What is the '{module_name}' module for?")
        module_answers.append(
            f"'{module_full_name}' module documents itself as follows: '{module_summary}'."
        )

    if not (module_exports := module_members.module_all_exports):
        module_questions.append(f"Tell me the public members of the '{module_full_name}' module.")
        module_answers.append(
            f"'{module_name}' module lacks any public member exported through '__all__'."
        )
    else:
        module_questions.append(f"Tell me the public members of the '{module_name}' module.")
        module_answers.append(
            " ".join(
                [
                    f"{module_full_name} publicly exports the following members using '__all__':",
                    " ".join(
                        f"{counter + 1}. {module_export}"
                        for counter, module_export in enumerate(module_exports)
                    ),
                ]
            )
        )

    return generate_dataset(module_questions, module_answers)


@pydantic.validate_call(validate_return=True)
def generate_enum_member_dataset(
    enum_member: str, member_type_details: EnumDetails
) -> list[Document]:
    member_questions = []
    member_answers = []

    member_questions.append(f"How many members are there in {enum_member}?")
    member_answers.append(f"{enum_member} has {len(member_type_details.enum_members)} members.")

    member_questions.append(f"What are the different members of {enum_member}?")
    member_answers.append(
        " ".join(
            [
                f"Different members of {enum_member} are as follows:",
                " ".join(
                    f"{counter + 1}. {enum_member.enum_member}"
                    for counter, enum_member in enumerate(member_type_details.enum_members)
                ),
            ]
        )
    )

    member_questions.append(f"List just the names of different members of {enum_member}.")
    member_answers.append(
        " ".join(
            [
                f"Different members of {enum_member} have the following names:",
                " ".join(
                    f"{counter + 1}. {enum_member.enum_member_name}"
                    for counter, enum_member in enumerate(member_type_details.enum_members)
                ),
            ]
        )
    )
    member_questions.append(f"Only show the different values supported by {enum_member}.")
    member_answers.append(
        " ".join(
            [
                "{enum_member} supports the following values:",
                " ".join(
                    f"{counter + 1}. {enum_member.enum_member_value}"
                    for counter, enum_member in enumerate(member_type_details.enum_members)
                ),
            ]
        )
    )

    return generate_dataset(member_questions, member_answers)


@pydantic.validate_call(validate_return=True)
def generate_class_member_dataset(  # noqa: C901, PLR0912, PLR0915
    class_member: str, member_type_details: ClassDetails
) -> list[Document]:
    member_questions = []
    member_answers = []

    if not (class_parameters := member_type_details.class_parameters):
        member_questions.append(f"What are the different parameters of {class_member}?")
        member_answers.append(f"{class_member} needs no arguments for instantiation.")
    else:
        member_questions.append(f"What are the different parameters of {class_member}?")
        member_answers.append(
            " ".join(
                [
                    f"{class_member} supports these arguments to initiate a new instance:",
                    " ".join(
                        f"{counter + 1}. {class_parameter.parameter_details}"
                        for counter, class_parameter in enumerate(class_parameters)
                    ),
                ]
            )
        )

    for class_parameter in class_parameters:
        parameter_name = class_parameter.parameter_name

        if (parameter_default := class_parameter.parameter_default) is inspect._empty:
            member_questions.append(f"Tell default value of '{parameter_name}' in {class_member}.")
            member_answers.append(f"'{parameter_name}' argument does not have a default value.")
        else:
            member_questions.append(f"Tell default value of '{parameter_name}' in {class_member}.")
            member_answers.append(
                f"Argument '{parameter_name}' takes {parameter_default} value by default."
            )

        if (parameter_annotation := class_parameter.parameter_annotation) is inspect._empty:
            member_questions.append(f"Name type hint for '{parameter_name}' in {class_member}.")
            member_answers.append(f"Parameter '{parameter_name}' does not have a type annotation.")
        else:
            member_questions.append(f"Name type hint for '{parameter_name}' in {class_member}.")
            member_answers.append(
                f"'parameter_name' parameter has '{parameter_annotation}' as type annotation."
            )

        if not (parameter_summary := class_parameter.parameter_summary):
            member_questions.append(f"What does '{parameter_name}' do in {class_member}?")
            member_answers.append(
                f"Docstring of {class_member} does not describe '{parameter_name}'."
            )
        else:
            member_questions.append(f"What does '{parameter_name}' do in {class_member}?")
            member_answers.append(
                f"{class_member} documents role of '{parameter_name}' as '{parameter_summary}'."
            )

    if not (class_methods := member_type_details.class_methods):
        member_questions.append(f"List names of the public methods of {class_member}.")
        member_answers.append(
            f"{class_member} does not have any public methods (not starting with '_')."
        )
    else:
        member_questions.append(f"List names of the public methods of {class_member}.")
        member_answers.append(
            " ".join(
                [
                    f"Here are the public methods of {class_member} (not starting with '_'):",
                    " ".join(
                        f"{counter + 1}. {class_method.method_name}"
                        for counter, class_method in enumerate(class_methods)
                    ),
                ]
            )
        )

    for class_method in class_methods:
        method_name = class_method.method_name

        if not (method_parameters := class_method.method_parameters):
            member_questions.append(
                f"What arguments do '{method_name}' method of {class_member} accept?"
            )
            member_answers.append(f"'{method_name}' method does not take any parameters.")
        else:
            member_questions.append(
                f"What arguments do '{method_name}' method of {class_member} accept?"
            )
            member_answers.append(
                " ".join(
                    [
                        f"'{method_name}' of {class_member} takes the following parameters:",
                        " ".join(
                            f"{counter + 1}. {method_parameter}"
                            for counter, method_parameter in enumerate(method_parameters)
                        ),
                    ]
                )
            )

        if not (method_summary := class_method.method_summary):
            member_questions.append(f"What does '{method_name}' method do in {class_member}?")
            member_answers.append(f"Docstring of '{method_name}' method is missing.")
        else:
            member_questions.append(f"What does '{method_name}' method do in {class_member}?")
            member_answers.append(f"Based on method docstring, its role is to '{method_summary}'.")

    if not (class_attributes := member_type_details.class_attributes):
        member_questions.append(f"Are there any public attributes of {class_member}?")
        member_answers.append(f"{class_member} has no public attributes (not starting with '_').")
    else:
        member_questions.append(f"Are there any public attributes of {class_member}?")
        member_answers.append(
            " ".join(
                [
                    f"These are the public (not starting with '_') attributes of {class_member}:",
                    " ".join(
                        f"{counter + 1}. {class_attribute.attribute_name}"
                        for counter, class_attribute in enumerate(class_attributes)
                    ),
                ]
            )
        )

    if not (class_summary := member_type_details.class_summary):
        member_questions.append(f"What does {class_member} do in short?")
        member_answers.append("Its docstring lacks a summary of its objective.")
    else:
        member_questions.append(f"What does {class_member} do in short?")
        member_answers.append(
            f"Based on documentation, objective of {class_member} is to: '{class_summary}'."
        )

    if not (class_notes := member_type_details.class_notes):
        member_questions.append(f"Mention any specific details for {class_member} to be aware of.")
        member_answers.append(f"Docstring of {class_member} does not note on specific details.")
    else:
        member_questions.append(f"Mention any specific details for {class_member} to be aware of.")
        member_answers.append(
            f"The {class_member} docstring highlights the following: '{class_notes}'."
        )

    return generate_dataset(member_questions, member_answers)


@pydantic.validate_call(validate_return=True)
def generate_function_member_dataset(  # noqa: C901, PLR0912, PLR0915
    function_member: str, member_type_details: FunctionDetails
) -> list[Document]:
    member_questions = []
    member_answers = []

    if not (function_parameters := member_type_details.function_parameters):
        member_questions.append(f"List various parameters of {function_member}.")
        member_answers.append(f"{function_member} does not take any parameters.")
    else:
        member_questions.append(f"List various parameters of {function_member}.")
        member_answers.append(
            " ".join(
                [
                    f"Different parameters of {function_member} are as follows:",
                    " ".join(
                        f"{counter + 1}. {function_parameter.parameter_details}"
                        for counter, function_parameter in enumerate(function_parameters)
                    ),
                ]
            )
        )

    for function_parameter in function_parameters:
        parameter_name = function_parameter.parameter_name

        if (parameter_default := function_parameter.parameter_default) is inspect._empty:
            member_questions.append(f"Default value of '{parameter_name}' in {function_member}?")
            member_answers.append(f"'{parameter_name}' argument does not have a default value.")
        else:
            member_questions.append(f"Default value of '{parameter_name}' in {function_member}?")
            member_answers.append(
                f"'{parameter_name}' parameter has default value of {parameter_default}."
            )

        if (parameter_annotation := function_parameter.parameter_annotation) is inspect._empty:
            member_questions.append(
                f"What is type annotation of '{parameter_name}' in {function_member}?"
            )
            member_answers.append(f"'{parameter_name}' parameter does not have a type annotation.")
        else:
            member_questions.append(
                f"What is type annotation of '{parameter_name}' in {function_member}?"
            )
            member_answers.append(
                f"Type annotation of '{parameter_name}' argument is '{parameter_annotation}'."
            )

        if not (parameter_summary := function_parameter.parameter_summary):
            member_questions.append(
                f"What is '{parameter_name}' parameter for in {function_member}?"
            )
            member_answers.append(f"Docstring of {function_member} lacks a description.")
        else:
            member_questions.append(
                f"What is '{parameter_name}' parameter for in {function_member}?"
            )
            member_answers.append(
                f"Based on {function_member} docstring, its role is '{parameter_summary}'."
            )

    if (
        returns_annotation := member_type_details.function_returns.returns_annotation
    ) is inspect._empty:
        member_questions.append(f"What is the return type annotation of {function_member}?")
        member_answers.append(
            f"{function_member} lacks a return type annotation. It may still return though."
        )
    else:
        member_questions.append(f"What is the return type annotation of {function_member}?")
        member_answers.append(
            f"Return type annotation for {function_member} is '{returns_annotation}'."
        )

    if not (returns_summary := member_type_details.function_returns.returns_summary):
        member_questions.append(f"What does {function_member} return?")
        member_answers.append(f"Docstring of {function_member} does not describe its return.")
    else:
        member_questions.append(f"What does {function_member} return?")
        member_answers.append(
            f"Based on {function_member} docstring, the return contains: '{returns_summary}'."
        )

    if not (function_summary := member_type_details.function_summary):
        member_questions.append(f"Summarise role of {function_member} in short.")
        member_answers.append(f"{function_member} docstring lacks a summary of its objective.")
    else:
        member_questions.append(f"Summarise role of {function_member} in short.")
        member_answers.append(
            f"Based on docstring, objective of {function_member} is to: '{function_summary}'."
        )

    if not (function_raises := member_type_details.function_raises):
        member_questions.append(f"Does {function_member} raise any specific exception?")
        member_answers.append(
            f"Docstring of {function_member} does not mention any specific exceptions."
        )
    else:
        member_questions.append(f"Does {function_member} raise any specific exception?")
        member_answers.append(
            " ".join(
                [
                    "Based on {function_member}'s docstring, it can raise the following:",
                    " ".join(
                        f"{counter + 1}. {function_raise.raises_details}"
                        for counter, function_raise in enumerate(function_raises)
                    ),
                ]
            )
        )

    if not (function_warns := member_type_details.function_warns):
        member_questions.append(f"Does {function_member} throw any specific warnings?")
        member_answers.append(
            f"{function_member}'s docstring lacks any mention of specific warnings."
        )
    else:
        member_questions.append(f"Does {function_member} throw any specific warnings?")
        member_answers.append(
            " ".join(
                [
                    f"Based on the docstring, {function_member} can throw the following warnings:",
                    " ".join(
                        f"{counter + 1}. {function_warn.warns_details}"
                        for counter, function_warn in enumerate(function_warns)
                    ),
                ]
            )
        )

    if not (function_notes := member_type_details.function_notes):
        member_questions.append(
            f"Is there any specific details for {function_member} to be aware of?"
        )
        member_answers.append(
            f"Docstring of {function_member} lacks any notes on specific details."
        )
    else:
        member_questions.append(
            f"Is there any specific details for {function_member} to be aware of?"
        )
        member_answers.append(
            f"The {function_member}'s docstring highlights the following: '{function_notes}'."
        )

    if not (function_references := member_type_details.function_references):
        member_questions.append(f"Is there any reference for {function_member}?")
        member_answers.append(f"Documentation for {function_member} contains no references.")
    else:
        member_questions.append(f"Is there any reference for {function_member}?")
        member_answers.append(f"The docstring links the following: '{function_references}'.")

    if not (function_examples := member_type_details.function_examples):
        member_questions.append(f"Is there any example for {function_member}?")
        member_answers.append(f"Docstring for {function_member} lacks any examples.")
    else:
        member_questions.append(f"Is there any example for {function_member}?")
        member_answers.append(
            f"Documentation of {function_member} contains these examples: '{function_examples}'."
        )

    return generate_dataset(member_questions, member_answers)


@pydantic.validate_call(validate_return=True)
def generate_member_dataset(member_details: MemberDetails) -> list[Document]:
    member_name = member_details.member_name
    member_full_name = member_details.member_qualified_name

    member_questions: list[str] = []
    member_answers: list[str] = []

    member_questions.append(f"What is the parent module of '{member_name}'?")
    member_answers.append(f"'{member_details.member_module}' is the name of its parent module.")

    member_questions.append(f"What is the full name of '{member_name}' member?")
    member_answers.append(f"'{member_full_name}' is its fully qualified name.")

    member_questions.append(f"What is the hierarchy of {member_name} member?")
    member_answers.append(
        " ".join(
            [
                f"The hierarchy of '{member_name}' member is as follows:",
                " ".join(
                    f"{counter + 1}. {node}"
                    for counter, node in enumerate(member_details.member_hierarchy)
                ),
            ]
        )
    )

    if not (member_docstring := member_details.member_docstring):
        member_questions.append(f"What is the documentation of '{member_full_name}' member?")
        member_answers.append(f"'{member_full_name}' member does not have any documentation.")
    else:
        member_questions.append(f"What does '{member_full_name}' member do?")
        member_answers.append(f"Its documentation is as follows: '{member_docstring}'.")

    member_dataset = generate_dataset(member_questions, member_answers)

    if (member_type_details := member_details.member_type_details) is None:
        return member_dataset

    member_type = member_type_details.member_type

    member_dataset.append(
        Document(
            question=f"What is the type of '{member_name}' member?",
            answer=f"'{member_full_name}' member is of '{member_type.value}' type.",
        )
    )

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
