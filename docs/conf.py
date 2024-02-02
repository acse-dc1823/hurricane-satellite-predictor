import sys
import os

# Assuming 'docs' is your current working directory,
# and 'storm_predict/models' is the directory you want to include.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__name__), '..', 'storm_predict', 'models')))

project = 'Day After Tomorrow'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme'
]
html_theme = "sphinx_rtd_theme"
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"
