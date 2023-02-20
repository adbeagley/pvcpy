"""Set up utitilies functions"""
from pydicom import uid
from .helpers import *


def _generate_valuerep_module():
    """Run script to auto-generate module"""
    from ..source import generate_valuerep
    generate_valuerep.run()


def _generate_keywords_module():
    """Run script to auto-generate module"""
    from ..source import generate_keywords
    generate_keywords.run()


try:
    from . import keywords
except ImportError:
    _generate_keywords_module()
    from . import keywords

try:
    from . import valuerep
except ImportError:
    _generate_valuerep_module()
    from . import valuerep
