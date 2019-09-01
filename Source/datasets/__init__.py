from __future__ import absolute_import
import warnings

from .synthtextlist import SynthTextList
from .synth90k import Synth90k
from .ic03 import IC03
from .iiit5k import IIIT5K
from .svt import SVT
from .ic13 import IC13
from .cute80 import CUTE80
from .svtp import SVTP
from .ic15 import IC15
from .totaltext import TOTALTEXT
from .ic19 import ic19
from .ic19_val import IC19VAL
from .prediction import prediction

__factory = {
    'synthtextlist': SynthTextList,
    'synth90k': Synth90k,
    'ic03': IC03,
    'iiit5k': IIIT5K,
    'svt': SVT,
    'ic13': IC13,
    'cute80': CUTE80,
    'svtp': SVTP,
    'ic15': IC15,
    'totaltext': TOTALTEXT,
    'ic19': ic19,
    'ic19_val': IC19VAL,
    'prediction':prediction
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
      The dataset name. Can be one of __factory
    root : str
      The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
