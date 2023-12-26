import enum
import importlib
import importlib.util
import inspect
import logging
import pkgutil
import types
import typing

import pydantic
from numpydoc.docscrape import NumpyDocString

from .utils_generation import (
    Attribute,
    ClassDetails,
    EnumDetails,
    EnumMember,
    FunctionDetails,
    MemberDetails,
    MemberType,
    Method,
    Module,
    ModuleMember,
    Package,
    Parameter,
    Raises,
    Returns,
    Warns,
)

LOGGER = logging.getLogger(__name__)


@pydantic.validate_call(validate_return=True)
def import_package(package_name: str) -> pydantic.InstanceOf[types.ModuleType]:
    package_spec = importlib.util.find_spec(package_name)

    if package_spec is None:
        LOGGER.error(f"spec for {package_name=} could not be found")

        raise ValueError(f"{package_name=} is not found")

    package = importlib.util.module_from_spec(package_spec)

    return package


@pydantic.validate_call(validate_return=True)
def get_all_package_contents(package_name: str) -> list[Package]:
    package_contents = []

    sub_packages_stack: list[tuple[str, str | None]] = [(package_name, None)]

    while sub_packages_stack:
        current_package_name, parent_package_name = sub_packages_stack.pop()

        current_package_hierarchy = current_package_name.split(".")

        try:
            current_package_loader = import_package(current_package_name)
        except ImportError:
            LOGGER.warning(f"{current_package_name=} could not be imported")

            continue

        try:
            current_package = importlib.import_module(current_package_name)
        except ImportError:
            LOGGER.warning(f"{current_package_name=} could not be imported")

            continue

        current_package_sub_packages = []
        current_package_modules = []

        for _, name, ispkg in pkgutil.walk_packages(
            path=current_package_loader.__path__, prefix=f"{current_package_loader.__name__}."
        ):
            if "tests" in name:
                continue

            if "." in name.removeprefix(f"{current_package_name}."):
                continue

            if ispkg:
                current_package_sub_packages.append(name)
            else:
                current_package_modules.append(name)

        package_contents.append(
            Package(
                package_name=current_package_hierarchy[-1],
                package_qualified_name=current_package_name,
                package_hierarchy=current_package_hierarchy,
                parent_package_name=parent_package_name,
                children_sub_packages_names=[
                    sub_package.removeprefix(f"{current_package_name}.")
                    for sub_package in current_package_sub_packages
                ],
                children_modules_names=[
                    module.removeprefix(f"{current_package_name}.")
                    for module in current_package_modules
                ],
                package_summary=getattr(current_package, "__doc__", None),
                package_all_exports=getattr(current_package, "__all__", None),
            )
        )

        for sub_package_name in current_package_sub_packages:
            sub_packages_stack.append((sub_package_name, current_package_name))  # noqa: PERF401

    return package_contents


@pydantic.validate_call(validate_return=True)
def get_all_module_contents(module_name: str) -> Module:
    module_hierarchy = module_name.split(".")

    module = importlib.import_module(module_name)

    module_contents = inspect.getmembers(
        module, predicate=lambda member: inspect.getmodule(member) == module
    )

    return Module(
        module_name=module_hierarchy[-1],
        module_qualified_name=module_name,
        module_hierarchy=module_hierarchy,
        package_name=".".join(module_hierarchy[:-1]),
        module_members=[
            ModuleMember(member_name=member[0], member_object=member[1])
            for member in module_contents
        ],
        module_summary=inspect.getdoc(module),
        module_all_exports=getattr(importlib.import_module(module_name), "__all__", None),
    )


@pydantic.validate_call(validate_return=True)
def get_all_parameters_details(
    signature: pydantic.InstanceOf[inspect.Signature],
    docstring: pydantic.InstanceOf[NumpyDocString],
) -> list[Parameter]:
    parameter_signature = {
        parameter.name: {
            "parameter_default": parameter.default,
            "parameter_annotation": parameter.annotation,
            "parameter_kind": parameter.kind.description,
        }
        for _, parameter in signature.parameters.items()
    }
    parameter_docstring = {
        parameter.name: {
            "parameter_annotation": parameter.type,
            "parameter_summary": " ".join(parameter.desc),
        }
        for parameter in docstring["Parameters"]
    }

    parameter_details = [
        Parameter.model_validate(
            {
                "parameter_name": parameter_name,
                "parameter_default": parameter_signature_details["parameter_default"],
                "parameter_annotation": parameter_docstring.get(parameter_name, {}).get(
                    "parameter_annotation", None
                )
                or parameter_signature_details["parameter_annotation"],
                "parameter_kind": parameter_signature_details["parameter_kind"],
                "parameter_summary": parameter_docstring.get(parameter_name, {}).get(
                    "parameter_summary", None
                ),
            }
        )
        for parameter_name, parameter_signature_details in parameter_signature.items()
    ]

    return parameter_details


@pydantic.validate_call(validate_return=True)
def get_all_returns_details(
    signature: pydantic.InstanceOf[inspect.Signature],
    docstring: pydantic.InstanceOf[NumpyDocString],
) -> Returns:
    returns_signature = signature.return_annotation

    if not docstring["Returns"]:
        return Returns(returns_annotation=returns_signature)

    returns_docstring = next(
        {"returns_annotation": returns.type, "returns_summary": " ".join(returns.desc)}
        for returns in docstring["Returns"]
    )

    return Returns(
        returns_annotation=returns_docstring.get("returns_annotation", None) or returns_signature,
        returns_summary=returns_docstring.get("returns_summary", None),
    )


@pydantic.validate_call(validate_return=True)
def get_all_member_details(
    module_name: str, member_name: str, member_object: typing.Any  # noqa: ANN401
) -> MemberDetails:
    member_hierarchy = [*module_name.split("."), member_name]

    member_details: dict[str, typing.Any] = {
        "member_name": member_name,
        "member_qualified_name": ".".join(member_hierarchy),
        "member_hierarchy": member_hierarchy,
        "member_module": member_hierarchy[-2],
    }

    member_details["member_docstring"] = inspect.getdoc(member_object) or ""
    parsed_docstring = NumpyDocString(member_details["member_docstring"])

    if isinstance(member_object, enum.EnumType):
        member_details["member_type_details"] = EnumDetails(
            member_type=MemberType.ENUM,
            enum_members=[
                EnumMember(enum_member_name=enum_member.name, enum_member_value=enum_member.value)
                for enum_member in member_object
            ],
        )
    elif inspect.isclass(member_object):
        member_details["member_type_details"] = ClassDetails(
            member_type=MemberType.CLASS,
            class_parameters=get_all_parameters_details(
                inspect.signature(member_object), parsed_docstring
            ),
            class_methods=[
                Method(
                    method_name=method[0],
                    method_parameters=[
                        parameter
                        for parameter, _ in inspect.signature(method[1]).parameters.items()
                    ],
                    method_summary=inspect.getdoc(method[1]),
                )
                for method in inspect.getmembers(member_object, predicate=inspect.ismethod)
                if not method[0].startswith("_")
            ],
            class_attributes=[
                Attribute(attribute_name=attribute[0])
                for attribute in inspect.getmembers(
                    member_object,
                    predicate=lambda member: not inspect.ismethod(member) and not callable(member),
                )
                if not attribute[0].startswith("_")
            ],
            class_summary=" ".join(
                parsed_docstring["Summary"] + parsed_docstring["Extended Summary"]
            ),
            class_notes=" ".join(parsed_docstring["See Also"] + parsed_docstring["Notes"]),
        )
    elif callable(member_object):
        member_details["member_type_details"] = FunctionDetails(
            member_type=MemberType.FUNCTION,
            function_parameters=get_all_parameters_details(
                inspect.signature(member_object), parsed_docstring
            ),
            function_returns=get_all_returns_details(
                inspect.signature(member_object), parsed_docstring
            ),
            function_summary=" ".join(
                parsed_docstring["Summary"] + parsed_docstring["Extended Summary"]
            ),
            function_raises=[
                Raises(raises_type=raises.type, raises_summary=" ".join(raises.desc))
                for raises in parsed_docstring["Raises"]
            ],
            function_warns=[
                Warns(warns_type=warns.type, warns_summary=" ".join(warns.desc))
                for warns in parsed_docstring["Warns"]
            ],
            function_notes="".join(parsed_docstring["Notes"]),
            function_references="".join(parsed_docstring["References"]),
            function_examples="".join(parsed_docstring["Examples"]),
        )

    return MemberDetails.model_validate(member_details)


__all__ = [
    "get_all_member_details",
    "get_all_module_contents",
    "get_all_package_contents",
    "get_all_parameters_details",
    "import_package",
]
