
# Do you need np in the tamoc namespace?
import numpy as np

# __version__ is injected into the __config__ file by meson
from .__config__ import __version__

# Set flag indicating whether error messages should be verbose
DEBUG = False

# Suppress numpy error messages unless running in DEBUG mode
if not DEBUG:
    np.seterr(all='ignore')
