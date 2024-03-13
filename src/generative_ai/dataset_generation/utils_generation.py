"""Define functionalities for type annotations in dataset generation step."""

import enum
import functools
import typing

import pydantic


class PackageDetails(pydantic.BaseModel):
    """Store details of a python package.

    Attributes
    ----------
    package_name : str
        name of the package
    package_qualified_name : str
        fully qualified name of the package that can be used to import the package
    package_hierarchy : list[str]
        hierarchy of the package, with respect to the root package, if any
    parent_package_name : str | None
        name of the package of which it is a sub-package, is any
    children_sub_packages_names : list[str]
        names of sub-packages in the package, if any
    children_modules_names : list[str]
        names of any modules in the package, if any
    package_summary : str | None, optional
        ``__doc__`` attribute of the package, if any, by default None
    package_all_exports : list[str] | None, optional
        publicly exported objects that can be imported from the package, if any, by default None
    """

    package_name: str
    package_qualified_name: str
    package_hierarchy: list[str]
    parent_package_name: str | None
    children_sub_packages_names: list[str]
    children_modules_names: list[str]
    package_summary: str | None = None
    package_all_exports: list[str] | None = None


class ModuleMemberDetails(pydantic.BaseModel):
    """Store details of an object of a python module.

    Attributes
    ----------
    member_name : str
        name of the object
    member_object : typing.Any
        original object
    """

    member_name: str
    member_object: typing.Any


class ModuleDetails(pydantic.BaseModel):
    """Store details of a python module.

    Attributes
    ----------
    module_name : str
        name of the module
    module_qualified_name : str
        fully qualified name of the module that can be used to import the module
    module_hierarchy : list[str]
        hierarchy of the module with respect to the root package
    package_name : str
        name of the package which contains the module
    module_members : list[ModuleMemberDetails]
        objects in the module
    module_summary : str | None, optional
        ``__doc__`` attribute of the module, if any, by default None
    module_all_exports : list[str] | None, optional
        publicly exported objects that can be imported from the module, if any, by default None
    """

    module_name: str
    module_qualified_name: str
    module_hierarchy: list[str]
    package_name: str
    module_members: list[ModuleMemberDetails]
    module_summary: str | None = None
    module_all_exports: list[str] | None = None


class MemberType(str, enum.Enum):
    """Define supported member types."""

    ENUM = "enum"
    CLASS = "class"
    FUNCTION = "function"


class EnumMemberDetails(pydantic.BaseModel):
    """Store details of an enum member.

    Attributes
    ----------
    enum_member_name : str
        name of the enum member
    enum_member_value : typing.Any
        value of the enum member
    """

    enum_member_name: str
    enum_member_value: typing.Any

    @pydantic.computed_field
    @functools.cached_property
    def enum_member(self: "EnumMemberDetails") -> str:
        """Store name and value of the enum member.

        Returns
        -------
        str
            name and value of the enum member
        """
        return f"{self.enum_member_name} (corresponding to '{self.enum_member_value}')"


class EnumDetails(pydantic.BaseModel):
    """Store details of an enum.

    Attributes
    ----------
    member_type : typing.Literal[MemberType.ENUM]
        type of the member
    enum_members : list[EnumMemberDetails]
        members of the enum
    """

    member_type: typing.Literal[MemberType.ENUM]
    enum_members: list[EnumMemberDetails]


class ParameterDetails(pydantic.BaseModel):
    """Store details of an argument of a class or a function.

    Attributes
    ----------
    parameter_name : str
        name of the parameter
    parameter_default : typing.Any
        default value of the parameter
    parameter_annotation : typing.Any
        type annotation of the parameter
    parameter_kind : str
        kind of argument, see Notes for possible values
    parameter_summary : str | None, optional
        argument description in class or function docstring, if any, by default None

    Notes
    -----
    * The kind must take a value from ``inspect._ParameterKind``:

        * positional-only
        * positional or keyword
        * variadic positional
        * keyword-only
        * variadic keyword
    """

    parameter_name: str
    parameter_default: typing.Any
    parameter_annotation: typing.Any
    parameter_kind: str
    parameter_summary: str | None = None

    @pydantic.computed_field
    @functools.cached_property
    def parameter_details(self: "ParameterDetails") -> str:
        """Store name and kind of the parameter.

        Returns
        -------
        str
            name and kind of the parameter
        """
        return f"'{self.parameter_name}', of type '{self.parameter_kind}'"


class MethodDetails(pydantic.BaseModel):
    """Store details of a method of a class.

    Attributes
    ----------
    method_name : str
        name of the method
    method_parameters : list[str]
        parameters of the method
    method_summary : str | None, optional
        ``__doc__`` attribute of the method, if any, by default None
    """

    method_name: str
    method_parameters: list[str]
    method_summary: str | None = None


class AttributeDetails(pydantic.BaseModel):
    """Store details of an attribute of a class.

    Attributes
    ----------
    attribute_name : str
        name of the attribute
    """

    attribute_name: str


class ClassDetails(pydantic.BaseModel):
    """Store details of a class.

    Attributes
    ----------
    member_type : typing.Literal[MemberType.CLASS]
        type of the member
    class_parameters : list[ParameterDetails]
        parameters of the class
    class_methods : list[MethodDetails]
        methods of the class
    class_attributes : list[AttributeDetails]
        attributes of the class
    class_summary : str | None, optional
        Summary and Extended Summary sections from the class docstring, if any, by default None
    class_notes : str | None, optional
        Notes and See Also sections from the class docstring, if any, by default None
    """

    member_type: typing.Literal[MemberType.CLASS]
    class_parameters: list[ParameterDetails]
    class_methods: list[MethodDetails]
    class_attributes: list[AttributeDetails]
    class_summary: str | None = None
    class_notes: str | None = None


class ReturnDetails(pydantic.BaseModel):
    """Store details of the return type of a function.

    Attributes
    ----------
    returns_annotation : typing.Any
        type annotation of the return
    returns_summary : str | None, optional
        description of return in function docstring, if any, by default None
    """

    returns_annotation: typing.Any
    returns_summary: str | None = None


class RaiseDetails(pydantic.BaseModel):
    """Store details of the exception raised by a function.

    Attributes
    ----------
    raises_type : str | None, optional
        type of the exception, if any, by default None
    raises_summary : str | None, optional
        description of exception in function docstring, if any, by default None
    """

    raises_type: str | None = None
    raises_summary: str | None = None

    @pydantic.computed_field
    @functools.cached_property
    def raises_details(self: "RaiseDetails") -> str:
        """Store type and summary of the exception.

        Returns
        -------
        str
            type and summary of the exception
        """
        return f"'{self.raises_type}' ('{self.raises_summary}')"


class WarnDetails(pydantic.BaseModel):
    """Store details of the warning raised by a function.

    Attributes
    ----------
    warns_type : str | None, optional
        type of the warning, if any, by default None
    warns_summary : str | None, optional
        description of warning in function docstring, if any, by default None
    """

    warns_type: str | None = None
    warns_summary: str | None = None

    @pydantic.computed_field
    @functools.cached_property
    def warns_details(self: "WarnDetails") -> str:
        """Store type and summary of the warning.

        Returns
        -------
        str
            type and summary of the warning
        """
        return f"'{self.warns_type}' ('{self.warns_summary}')"


class FunctionDetails(pydantic.BaseModel):
    """Store details of a function.

    Attributes
    ----------
    member_type : typing.Literal[MemberType.FUNCTION]
        type of the member
    function_parameters : list[ParameterDetails]
        parameters of the function
    function_returns : Returns
        return type of the function
    function_summary : str | None, optional
        Summary and Extended Summary sections from the function docstring, if any, by default None
    function_raises : list[Raises] | None, optional
        exceptions raised by the function, if any, by default None
    function_warns : list[Warns] | None, optional
        warnings raised by the function, if any, by default None
    function_notes : str | None, optional
        Notes and See Also sections from the function docstring, if any, by default None
    function_references : str | None, optional
        References section from the function docstring, if any, by default None
    function_examples : str | None, optional
        Examples section from the function docstring, if any, by default None
    """

    member_type: typing.Literal[MemberType.FUNCTION]
    function_parameters: list[ParameterDetails]
    function_returns: ReturnDetails
    function_summary: str | None = None
    function_raises: list[RaiseDetails] | None = None
    function_warns: list[WarnDetails] | None = None
    function_notes: str | None = None
    function_references: str | None = None
    function_examples: str | None = None


class MemberDetails(pydantic.BaseModel):
    """Store details of a member of a module.

    Attributes
    ----------
    member_name : str
        name of the member
    member_qualified_name : str
        fully qualified name of the member that can be used to import the member
    member_hierarchy : list[str]
        hierarchy of the member with respect to the root package
    member_module : str
        name of the module which contains the member
    member_docstring : str
        ``__doc__`` attribute of the member, if any
    member_type_details : EnumDetails | ClassDetails | FunctionDetails | None, optional
        details of the member, if any, by default None
    """

    member_name: str
    member_qualified_name: str
    member_hierarchy: list[str]
    member_module: str
    member_docstring: str
    member_type_details: EnumDetails | ClassDetails | FunctionDetails | None = pydantic.Field(
        default=None, discriminator="member_type"
    )


class SplitName(str, enum.Enum):
    """Define supported split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitProportions(pydantic.BaseModel):
    """Specify proportions of train, validation and test subsets.

    Attributes
    ----------
    train_proportion : float
        proportion of train subset, must be in (0, 1)
    validation_proportion : float
        proportion of validation subset, must be in (0, 1)
    test_proportion : float
        proportion of test subset, must be in (0, 1)

    Notes
    -----
    * The proportions must add up to 1.
    """

    train_proportion: typing.Annotated[float, pydantic.Field(gt=0, lt=1)]
    validation_proportion: typing.Annotated[float, pydantic.Field(gt=0, lt=1)]
    test_proportion: typing.Annotated[float, pydantic.Field(gt=0, lt=1)]

    @pydantic.model_validator(mode="after")
    def validate_proportions(self: "SplitProportions") -> "SplitProportions":
        """Ensure that specified proportions add up to 1.

        Returns
        -------
        SplitProportions
            instance of the class itself

        Raises
        ------
        ValueError
            if specified proportions do not sum up to 1
        """
        if self.train_proportion + self.validation_proportion + self.test_proportion != 1:
            raise ValueError("Proportions must sum up to 1.")

        return self


class Document(pydantic.BaseModel):
    """Store details of a document.

    Attributes
    ----------
    context : str
        details containing the description
    question : str
        question to be answered or instructions to follow using the ``context``
    answer : str
        answer to the question or instruction based on the ``context``
    """

    context: str
    question: str
    answer: str
    split: SplitName


class Dataset(pydantic.BaseModel):
    """Store details of a dataset.

    Attributes
    ----------
    retrieval_chunks : list[str]
        chunks of text to be used for retrieval
    tuning_triplets : list[tuple[str, str, SplitName]]
        pairs of question and answer to be used for tuning and their split allocation
    """

    retrieval_chunks: list[str]
    tuning_documents: list[Document]


class JSONDocument(pydantic.BaseModel):
    """Store details of a document in JSON format.

    Attributes
    ----------
    context : str
        details containing the description
    question : str
        question to be answered or instructions to follow using the ``context``
    answer : str
        answer to the question or instruction based on the ``context``
    split : SplitName
        split allocation of the document
    """

    context: str
    question: str
    answer: str
    split: SplitName


class JSONDataset(pydantic.BaseModel):
    """Store all details for querying a package documentation in JSON format.

    Attributes
    ----------
    retrieval_documents : list[str]
        chunks of text to be used for retrieval
    tuning_documents : list[JSONDocument]
        pairs of question and answer to be used for tuning
    """

    retrieval_documents: list[str]
    tuning_documents: list[JSONDocument]


__all__ = [
    "AttributeDetails",
    "ClassDetails",
    "Dataset",
    "Document",
    "EnumDetails",
    "EnumMemberDetails",
    "FunctionDetails",
    "JSONDataset",
    "JSONDocument",
    "MemberDetails",
    "MemberType",
    "MethodDetails",
    "ModuleDetails",
    "ModuleMemberDetails",
    "PackageDetails",
    "ParameterDetails",
    "RaiseDetails",
    "ReturnDetails",
    "SplitName",
    "SplitProportions",
    "WarnDetails",
]
