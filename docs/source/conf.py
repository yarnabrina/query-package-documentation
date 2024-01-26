"""Configure Sphinx documentation."""
# pylint: disable=invalid-name
import sys

import generative_ai

sys.path.insert(0, "../src")

project = "query-package-documentation"
version = str(generative_ai.__version__)
project_copyright = "2023-2024, Anirban Ray"
author = "Anirban Ray"
release = f"v{version}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

smartquotes = False
today_fmt = "%Y-%m-%d"
highlight_language = "python3"
pygments_style = "friendly"
add_function_parentheses = False
add_module_names = False
option_emphasise_placeholders = True

html_theme = "furo"
html_theme_options = {"top_of_page_button": None}
html_title = f"{project} {release}"

html_last_updated_fmt = "%B %d, %Y"
html_use_index = True
html_split_index = False
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_output_encoding = "utf-8"

python_display_short_literal_types = True

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autoclass_content = "class"
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "langchain": ("https://api.python.langchain.com/en/stable", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/stable", None),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
