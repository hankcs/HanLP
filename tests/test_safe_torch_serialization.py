import copyreg
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hanlp.layers.embeddings.contextual_string_embedding import RNNLanguageModel
from hanlp.utils.torch_serialization import load_tensor_checkpoint


class _Payload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return eval, ("__import__('pathlib').Path({!r}).touch()".format(self.marker),)


class TestSafeTorchSerialization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_existing_rnn_checkpoint_formats(self):
        model = RNNLanguageModel(4, True, 3, 2)
        path = str(self.root / 'model.pt')
        model.save(path)
        self.assert_model_equal(model, RNNLanguageModel.load_language_model(path))
        state = load_tensor_checkpoint(path)
        for protocol in (2, 4):
            for use_zip in (False, True):
                with self.subTest(protocol=protocol, use_zip=use_zip):
                    torch.save(state, path, pickle_protocol=protocol, _use_new_zipfile_serialization=use_zip)
                    self.assert_model_equal(model, RNNLanguageModel.load_language_model(path))

    def assert_model_equal(self, expected, actual):
        self.assertEqual(expected.is_forward_lm, actual.is_forward_lm)
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[key]), key)

    def test_checkpoint_blocks_pickle_execution(self):
        marker = self.root / 'executed'
        path = str(self.root / 'model.pt')
        for protocol in (2, 4):
            for use_zip in (False, True):
                with self.subTest(protocol=protocol, use_zip=use_zip):
                    torch.save({'nested': _Payload(marker)}, path, pickle_protocol=protocol,
                               _use_new_zipfile_serialization=use_zip)
                    with self.assertRaises(pickle.UnpicklingError):
                        RNNLanguageModel.load_language_model(path)
                    self.assertFalse(marker.exists())

    def test_checkpoint_does_not_dispatch_torchscript(self):
        path = str(self.root / 'script.pt')
        torch.jit.trace(torch.nn.Identity(), torch.zeros(1)).save(path)
        with patch.object(torch.jit, 'load') as jit_load:
            with self.assertRaises(pickle.UnpicklingError):
                load_tensor_checkpoint(path)
            jit_load.assert_not_called()

    def test_checkpoint_blocks_cached_pickle_extensions(self):
        marker = self.root / 'executed'
        path = str(self.root / 'model.pt')
        copyreg.add_extension('builtins', 'eval', 192)
        try:
            # A previously cached extension bypasses the C unpickler's
            # find_class method, so test with a populated process-wide cache.
            pickle.loads(pickle.dumps(eval, protocol=2))
            torch.save(_Payload(marker), path)
            with self.assertRaises(pickle.UnpicklingError):
                load_tensor_checkpoint(path)
            self.assertFalse(marker.exists())
        finally:
            copyreg.remove_extension('builtins', 'eval', 192)

    def test_checkpoint_does_not_use_torch_load(self):
        model = RNNLanguageModel(4, True, 3, 2)
        path = str(self.root / 'model.pt')
        model.save(path)
        # Some older torch.load versions ignore pickle_module altogether.
        with patch.object(torch, 'load', side_effect=AssertionError('Unsafe dispatch to torch.load')):
            self.assert_model_equal(model, RNNLanguageModel.load_language_model(path))

if __name__ == '__main__':
    unittest.main()
