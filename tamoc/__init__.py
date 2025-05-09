__version__ = "4.0.0"

import numpy as np

# Set flag indicating whether error messages should be verbose
DEBUG = False

# Suppress numpy error messages unless running in DEBUG mode
if not DEBUG:
    np.seterr(all='ignore')