from .gpumap_ import GPUMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba
numba.config.THREADING_LAYER = 'workqueue'

import pkg_resources

__version__ = pkg_resources.get_distribution("gpumap").version
