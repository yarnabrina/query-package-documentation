import importlib.resources
import json
import typing


class PackageMetadata(typing.TypedDict):
    Name: str
    Version: str
    Description: str
    Keywords: list[str]
    License: str
    Maintainers: list[str]
    Authors: list[str]
    Links: dict[str, str]


METADATA_CONTENTS: str = (
    importlib.resources.files("generative_ai").joinpath("metadata.json").read_text()
)
METADATA: PackageMetadata = json.loads(METADATA_CONTENTS)

__version__: str = METADATA["Version"]
__all__: list[str] = ["METADATA", "__version__"]
