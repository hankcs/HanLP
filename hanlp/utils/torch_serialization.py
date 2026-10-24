"""Read tensor checkpoints without allowing pickle to import arbitrary globals.

Unlike PyTorch's weights_only reader, the stdlib unpickler understands the
protocol 4 files saved by historical RNNLanguageModel versions. Supplying this
restricted pickle module also works with PyTorch versions predating weights_only.
This reader is for dense state dictionaries and primitive model configuration,
not pickled modules or TorchScript programs.
"""
import collections
import pickle
import sys

import torch


class Unpickler(pickle._Unpickler):
    # Use the Python implementation so extension opcodes cannot bypass
    # find_class through pickle's process-wide extension cache.
    def get_extension(self, code):
        raise pickle.UnpicklingError('Pickle extensions are not supported in tensor checkpoints')

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
            return getattr(torch._utils, name)
        # Legacy torch.save streams use these globals for persistent storages.
        # Only resolve actual storage classes from the installed torch module.
        if module in ('torch', 'torch.cuda') and name.endswith('Storage'):
            storage_module = torch if module == 'torch' else torch.cuda
            storage = getattr(storage_module, name, None)
            if storage in torch._storage_classes:
                return storage
        raise pickle.UnpicklingError('Unsupported checkpoint global: {}.{}'.format(module, name))


def load(file, **kwargs):
    """pickle.load interface used for the headers of legacy torch.save streams."""
    return Unpickler(file, **kwargs).load()


def load_tensor_checkpoint(path, map_location=None):
    """Load existing tensor checkpoints using an explicit global allowlist."""
    # PyTorch 1.13.1's torch.load discards an explicitly supplied pickle_module.
    # Use the format readers (available since 1.6) directly so every supported
    # version uses our restricted Unpickler, without changing global torch state.
    serialization = torch.serialization
    pickle_module = sys.modules[__name__]
    with open(path, 'rb') as file:
        if serialization._is_zipfile(file):
            with serialization._open_zipfile_reader(file) as archive:
                # Never dispatch checkpoint inputs to torch.jit.load.
                if serialization._is_torchscript_zip(archive):
                    raise pickle.UnpicklingError('Expected a tensor checkpoint, not TorchScript')
                return serialization._load(archive, map_location, pickle_module, encoding='utf-8')
        return serialization._legacy_load(file, map_location, pickle_module, encoding='utf-8')
