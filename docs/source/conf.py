# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Multi-Input Medical Image ML Toolkit'
copyright = '2024, Matt Leming'
author = 'Matt Leming'
release = '0.0.17'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
#    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_favicon = 'favicon.ico'

import sys,os
#wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#sys.path.insert(0,wd)
#sys.path.insert(0,os.path.join(wd,'src'))
#sys.path.insert(0,os.path.join(wd,'src','multi_med_image_ml'))

f = os.path.abspath(os.path.join('..', '..','src'))
sys.path.insert(0, f)
import multi_med_image_ml
release = multi_med_image_ml.__version__
