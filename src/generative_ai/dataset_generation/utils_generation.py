import enum
import functools
import typing

import pydantic


class Package(pydantic.BaseModel):
    package_name: str
    package_qualified_name: str
    package_hierarchy: list[str]
    parent_package_name: str | None
    children_sub_packages_names: list[str]
    children_modules_names: list[str]
    package_summary: str | None = None
    package_all_exports: list[str] | None = None


class ModuleMember(pydantic.BaseModel):
    member_name: str
    member_object: typing.Any


class Module(pydantic.BaseModel):
    module_name: str
    module_qualified_name: str
    module_hierarchy: list[str]
    package_name: str
    module_members: list[ModuleMember]
    module_summary: str | None = None
    module_all_exports: list[str] | None = None


class MemberType(str, enum.Enum):
    ENUM = "enum"
    CLASS = "class"
    FUNCTION = "function"


class EnumMember(pydantic.BaseModel):
    enum_member_name: str
    enum_member_value: typing.Any

    @pydantic.computed_field
    @functools.cached_property
    def enum_member(self: "EnumMember") -> str:
        return f"{self.enum_member_name} (corresponding to '{self.enum_member_value}')"


class EnumDetails(pydantic.BaseModel):
    member_type: typing.Literal[MemberType.ENUM]
    enum_members: list[EnumMember]


class Parameter(pydantic.BaseModel):
    parameter_name: str
    parameter_default: typing.Any
    parameter_annotation: typing.Any
    parameter_kind: str
    parameter_summary: str | None = None

    @pydantic.computed_field
    @functools.cached_property
    def parameter_details(self: "Parameter") -> str:
        return f"'{self.parameter_name}', of type '{self.parameter_kind}'"


class Method(pydantic.BaseModel):
    method_name: str
    method_parameters: list[str]
    method_summary: str | None = None


class Attribute(pydantic.BaseModel):
    attribute_name: str


class ClassDetails(pydantic.BaseModel):
    member_type: typing.Literal[MemberType.CLASS]
    class_parameters: list[Parameter]
    class_methods: list[Method]
    class_attributes: list[Attribute]
    class_summary: str | None = None
    class_notes: str | None = None


class Returns(pydantic.BaseModel):
    returns_annotation: typing.Any
    returns_summary: str | None = None


class Raises(pydantic.BaseModel):
    raises_type: str | None = None
    raises_summary: str | None = None

    @pydantic.computed_field
    @functools.cached_property
    def raises_details(self: "Raises") -> str:
        return f"'{self.raises_type}' ('{self.raises_summary}')"


class Warns(pydantic.BaseModel):
    warns_type: str | None = None
    warns_summary: str | None = None

    @pydantic.computed_field
    @functools.cached_property
    def warns_details(self: "Warns") -> str:
        return f"'{self.warns_type}' ('{self.warns_summary}')"


class FunctionDetails(pydantic.BaseModel):
    member_type: typing.Literal[MemberType.FUNCTION]
    function_parameters: list[Parameter]
    function_returns: Returns
    function_summary: str | None = None
    function_raises: list[Raises] | None = None
    function_warns: list[Warns] | None = None
    function_notes: str | None = None
    function_references: str | None = None
    function_examples: str | None = None


class MemberDetails(pydantic.BaseModel):
    member_name: str
    member_qualified_name: str
    member_hierarchy: list[str]
    member_module: str
    member_docstring: str
    member_type_details: EnumDetails | ClassDetails | FunctionDetails | None = pydantic.Field(
        default=None, discriminator="member_type"
    )


class Document(pydantic.BaseModel):
    context: str
    question: str
    answer: str

    @pydantic.computed_field
    @functools.cached_property
    def instruction_with_context(self: "Document") -> str:
        system_instruction = (
            "Below is a question that can be answered using the following context. "
            "Write an answer for the question appropriately without using any additional data."
        )

        return " ".join(
            [
                "<s>",
                f"[INST] {system_instruction} [/INST]",
                f"[INST] Context: {self.context} [/INST]",
                f"[INST] Question: {self.question} [/INST]",
                f"[INST] Answer: {self.answer} [/INST]",
                "</s>",
            ]
        )

    @pydantic.computed_field
    @functools.cached_property
    def instruction_without_context(self: "Document") -> str:
        return f"<s>[INST] {self.question} [/INST] {self.answer} </s>"


class Dataset(pydantic.BaseModel):
    retrieval_chunks: list[str]
    tuning_pairs: list[tuple[str, str]]

    @pydantic.computed_field
    @functools.cached_property
    def tuning_documents(self: "Dataset") -> list[Document]:
        return [
            Document(context=" ".join(self.retrieval_chunks), question=question, answer=answer)
            for question, answer in self.tuning_pairs
        ]


class JSONDocument(pydantic.BaseModel):
    context: str
    question: str
    answer: str
    instruction_with_context: str
    instruction_without_context: str


class JSONDataset(pydantic.BaseModel):
    retrieval_documents: list[str]
    tuning_documents: list[JSONDocument]


__all__ = [
    "Attribute",
    "ClassDetails",
    "Dataset",
    "Document",
    "EnumDetails",
    "EnumMember",
    "FunctionDetails",
    "JSONDataset",
    "JSONDocument",
    "MemberDetails",
    "MemberType",
    "Method",
    "Module",
    "ModuleMember",
    "Package",
    "Parameter",
    "Raises",
    "Returns",
    "Warns",
]
