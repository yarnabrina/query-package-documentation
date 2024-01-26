"""Define the package metadata and version."""

import importlib.resources

import pydantic


class PackageMetadata(pydantic.BaseModel):
    """Define keys and types of corresponding values for package metadata.

    Attributes
    ----------
    Name : str
        name of the package
    Version : str
        version of the package
    Description : str
        description of the package
    Keywords : list[str]
        keywords associated with the package
    License : str
        license of the package
    Maintainers : list[str]
        maintainers of the package
    Authors : list[str]
        authors of the package
    Links : dict[str, pydantic.HttpUrl]
        links associated with the package
    """

    Name: str
    Version: str
    Description: str
    Keywords: list[str]
    License: str
    Maintainers: list[str]
    Authors: list[str]
    Links: dict[str, pydantic.HttpUrl]


METADATA_CONTENTS: str = (
    importlib.resources.files("generative_ai").joinpath("metadata.json").read_text()
)
METADATA = PackageMetadata.model_validate_json(METADATA_CONTENTS)

__version__: str = METADATA.Version
__all__: list[str] = ["METADATA", "__version__"]
