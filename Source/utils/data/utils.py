import numpy as np
import warnings
import sys

from examples.config import get_args

global_args = get_args(sys.argv[1:])

warnings.simplefilter('ignore', np.RankWarning)

BANNED = ("'", '"', ',', '.')
