"""Read historical embedding caches without importing arbitrary pickle globals.

Keep this separate from the general pickle helper: embedding caches contain only
primitive vocabulary data and NumPy arrays, not arbitrary Python objects.
"""
import codecs
import importlib
import pickle

import numpy as np


def _decode_numpy_bytes(value, encoding):
    # NumPy pickles made with protocol 2 represent raw bytes this way.
    if not isinstance(value, str) or encoding != 'latin1':
        raise pickle.UnpicklingError('Unsupported NumPy byte encoding')
    return codecs.encode(value, encoding)


class _EmbeddingUnpickler(pickle.Unpickler):
    def __init__(self, file, numpy_arrays=False):
        super().__init__(file)
        self.numpy_arrays = numpy_arrays

    def find_class(self, module, name):
        if self.numpy_arrays:
            if module == 'numpy' and name in ('ndarray', 'dtype'):
                return getattr(np, name)
            if module == '_codecs' and name == 'encode':
                return _decode_numpy_bytes
            # NumPy 2 renamed core to _core. Accept caches from either version,
            # resolving to the installed version rather than importing the input.
            core = 'numpy._core' if hasattr(np, '_core') else 'numpy.core'
            if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
                return importlib.import_module(core + '.multiarray')._reconstruct
            if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
                return importlib.import_module(core + '.numeric')._frombuffer
        raise pickle.UnpicklingError('Unsupported embedding cache global: {}.{}'.format(module, name))


def load_embedding_pickle(path, numpy_arrays=False):
    with open(path, 'rb') as src:
        return _EmbeddingUnpickler(src, numpy_arrays=numpy_arrays).load()
