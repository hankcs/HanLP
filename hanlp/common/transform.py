# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-03 14:44
import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple, Union, List

from hanlp_common.constant import EOS, PAD
from hanlp_common.structure import SerializableDict
from hanlp_common.configurable import Configurable
from hanlp.common.vocab import Vocab
from hanlp.utils.io_util import get_resource
from hanlp_common.io import load_json
from hanlp_common.reflection import classpath_of, str_to_type
from hanlp.utils.string_util import ispunct


class ToIndex(ABC):

    def __init__(self, vocab: Vocab = None) -> None:
        super().__init__()
        if vocab is None:
            vocab = Vocab()
        self.vocab = vocab

    @abstractmethod
    def __call__(self, sample):
        pass

    def save_vocab(self, save_dir, filename='vocab.json'):
        vocab = SerializableDict()
        vocab.update(self.vocab.to_dict())
        vocab.save_json(os.path.join(save_dir, filename))

    def load_vocab(self, save_dir, filename='vocab.json'):
        save_dir = get_resource(save_dir)
        vocab = SerializableDict()
        vocab.load_json(os.path.join(save_dir, filename))
        self.vocab.copy_from(vocab)


class FieldToIndex(ToIndex):

    def __init__(self, src, vocab: Vocab, dst=None) -> None:
        super().__init__(vocab)
        self.src = src
        if not dst:
            dst = f'{src}_id'
        self.dst = dst

    def __call__(self, sample: dict):
        sample[self.dst] = self.vocab(sample[self.src])
        return sample

    def save_vocab(self, save_dir, filename=None):
        if not filename:
            filename = f'{self.dst}_vocab.json'
        super().save_vocab(save_dir, filename)

    def load_vocab(self, save_dir, filename=None):
        if not filename:
            filename = f'{self.dst}_vocab.json'
        super().load_vocab(save_dir, filename)


class VocabList(list):

    def __init__(self, *fields) -> None:
        super().__init__()
        for each in fields:
            self.append(FieldToIndex(each))

    def append(self, item: Union[str, Tuple[str, Vocab], Tuple[str, str, Vocab], FieldToIndex]) -> None:
        if isinstance(item, str):
            item = FieldToIndex(item)
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                item = FieldToIndex(src=item[0], vocab=item[1])
            elif len(item) == 3:
                item = FieldToIndex(src=item[0], dst=item[1], vocab=item[2])
            else:
                raise ValueError(f'Unsupported argument length: {item}')
        elif isinstance(item, FieldToIndex):
            pass
        else:
            raise ValueError(f'Unsupported argument type: {item}')
        super(self).append(item)

    def save_vocab(self, save_dir):
        for each in self:
            each.save_vocab(save_dir, None)

    def load_vocab(self, save_dir):
        for each in self:
            each.load_vocab(save_dir, None)


class VocabDict(SerializableDict):

    def __init__(self, *args, **kwargs) -> None:
        """A dict holding :class:`hanlp.common.vocab.Vocab` instances. When used as a transform, it transforms the field
        corresponding to each :class:`hanlp.common.vocab.Vocab` into indices.

        Args:
            *args: A list of vocab names.
            **kwargs: Names and corresponding :class:`hanlp.common.vocab.Vocab` instances.
        """
        vocabs = dict(kwargs)
        for each in args:
            vocabs[each] = Vocab()
        super().__init__(vocabs)

    def save_vocabs(self, save_dir, filename='vocabs.json'):
        """Save vocabularies to a directory.

        Args:
            save_dir: The directory to save vocabularies.
            filename:  The name for vocabularies.
        """
        vocabs = SerializableDict()
        for key, value in self.items():
            if isinstance(value, Vocab):
                vocabs[key] = value.to_dict()
        vocabs.save_json(os.path.join(save_dir, filename))

    def load_vocabs(self, save_dir, filename='vocabs.json', vocab_cls=Vocab):
        """Load vocabularies from a directory.

        Args:
            save_dir: The directory to load vocabularies.
            filename:  The name for vocabularies.
        """
        save_dir = get_resource(save_dir)
        vocabs = SerializableDict()
        vocabs.load_json(os.path.join(save_dir, filename))
        self._load_vocabs(self, vocabs, vocab_cls)

    @staticmethod
    def _load_vocabs(vd, vocabs: dict, vocab_cls=Vocab):
        """

        Args:
            vd:
            vocabs:
            vocab_cls: Default class for the new vocab
        """
        for key, value in vocabs.items():
            if 'idx_to_token' in value:
                cls = value.get('type', None)
                if cls:
                    cls = str_to_type(cls)
                else:
                    cls = vocab_cls
                vocab = cls()
                vocab.copy_from(value)
                vd[key] = vocab
            else:  # nested Vocab
                # noinspection PyTypeChecker
                vd[key] = nested = VocabDict()
                VocabDict._load_vocabs(nested, value, vocab_cls)

    def lock(self):
        """
        Lock each vocabs.
        """
        for key, value in self.items():
            if isinstance(value, Vocab):
                value.lock()

    @property
    def mutable(self):
        status = [v.mutable for v in self.values() if isinstance(v, Vocab)]
        return len(status) == 0 or any(status)

    def __call__(self, sample: dict):
        for key, value in self.items():
            if isinstance(value, Vocab):
                field = sample.get(key, None)
                if field is not None:
                    sample[f'{key}_id'] = value(field)
        return sample

    def __getattr__(self, key):
        if key.startswith('__'):
            return dict.__getattr__(key)
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __getitem__(self, k: str) -> Vocab:
        return super().__getitem__(k)

    def __setitem__(self, k: str, v: Vocab) -> None:
        super().__setitem__(k, v)

    def summary(self, logger: logging.Logger = None):
        """Log a summary of vocabs using a given logger.

        Args:
            logger: The logger to use.
        """
        for key, value in self.items():
            if isinstance(value, Vocab):
                report = value.summary(verbose=False)
                if logger:
                    logger.info(f'{key}{report}')
                else:
                    print(f'{key}{report}')

    def put(self, **kwargs):
        """Put names and corresponding :class:`hanlp.common.vocab.Vocab` instances into self.

        Args:
            **kwargs: Names and corresponding :class:`hanlp.common.vocab.Vocab` instances.
        """
        for k, v in kwargs.items():
            self[k] = v


class NamedTransform(ABC):
    def __init__(self, src: str, dst: str = None) -> None:
        if dst is None:
            dst = src
        self.dst = dst
        self.src = src

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        return sample


class ConfigurableTransform(Configurable, ABC):
    @property
    def config(self):
        return dict([('classpath', classpath_of(self))] +
                    [(k, v) for k, v in self.__dict__.items() if not k.startswith('_')])

    @classmethod
    def from_config(cls, config: dict):
        """

        Args:
          config: 
          kwargs: 
          config: dict: 

        Returns:

        
        """
        cls = config.get('classpath', None)
        assert cls, f'{config} doesn\'t contain classpath field'
        cls = str_to_type(cls)
        config = dict(config)
        config.pop('classpath')
        return cls(**config)


class ConfigurableNamedTransform(NamedTransform, ConfigurableTransform, ABC):
    pass


class EmbeddingNamedTransform(ConfigurableNamedTransform, ABC):

    def __init__(self, output_dim: int, src: str, dst: str) -> None:
        super().__init__(src, dst)
        self.output_dim = output_dim


class RenameField(NamedTransform):

    def __call__(self, sample: dict):
        sample[self.dst] = sample.pop(self.src)
        return sample


class CopyField(object):
    def __init__(self, src, dst) -> None:
        self.dst = dst
        self.src = src

    def __call__(self, sample: dict) -> dict:
        sample[self.dst] = sample[self.src]
        return sample


class FilterField(object):
    def __init__(self, *keys) -> None:
        self.keys = keys

    def __call__(self, sample: dict):
        sample = dict((k, sample[k]) for k in self.keys)
        return sample


class TransformList(list):
    """Composes several transforms together.

    Args:
      transforms(list of ``Transform`` objects): list of transforms to compose.
    Example:

    Returns:

    >>> transforms.TransformList(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> )
    """

    def __init__(self, *transforms) -> None:
        super().__init__()
        self.extend(transforms)

    def __call__(self, sample):
        for t in self:
            sample = t(sample)
        return sample

    def index_by_type(self, t):
        for i, trans in enumerate(self):
            if isinstance(trans, t):
                return i


class LowerCase(object):
    def __init__(self, src, dst=None) -> None:
        if dst is None:
            dst = src
        self.src = src
        self.dst = dst

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            sample[self.dst] = src.lower()
        elif isinstance(src, list):
            sample[self.dst] = [x.lower() for x in src]
        return sample


class LowerCase3D(LowerCase):

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        sample[self.dst] = [[y.lower() for y in x] for x in src]
        return sample


class ToChar(object):
    def __init__(self, src, dst='char', max_word_length=None, min_word_length=None, pad=PAD) -> None:
        if dst is None:
            dst = src
        self.src = src
        self.dst = dst
        self.max_word_length = max_word_length
        self.min_word_length = min_word_length
        self.pad = pad

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            sample[self.dst] = self.to_chars(src)
        elif isinstance(src, list):
            sample[self.dst] = [self.to_chars(x) for x in src]
        return sample

    def to_chars(self, word: str):
        chars = list(word)
        if self.min_word_length and len(chars) < self.min_word_length:
            chars = chars + [self.pad] * (self.min_word_length - len(chars))
        if self.max_word_length:
            chars = chars[:self.max_word_length]
        return chars


class AppendEOS(NamedTransform):

    def __init__(self, src: str, dst: str = None, eos=EOS) -> None:
        super().__init__(src, dst)
        self.eos = eos

    def __call__(self, sample: dict) -> dict:
        sample[self.dst] = sample[self.src] + [self.eos]
        return sample


class WhitespaceTokenizer(NamedTransform):

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            sample[self.dst] = self.tokenize(src)
        elif isinstance(src, list):
            sample[self.dst] = [self.tokenize(x) for x in src]
        return sample

    @staticmethod
    def tokenize(text: str):
        return text.split()


class NormalizeDigit(object):
    def __init__(self, src, dst=None) -> None:
        if dst is None:
            dst = src
        self.src = src
        self.dst = dst

    @staticmethod
    def transform(word: str):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            sample[self.dst] = self.transform(src)
        elif isinstance(src, list):
            sample[self.dst] = [self.transform(x) for x in src]
        return sample


class Bigram(NamedTransform):

    def __init__(self, src: str, dst: str = None) -> None:
        if not dst:
            dst = f'{src}_bigram'
        super().__init__(src, dst)

    def __call__(self, sample: dict) -> dict:
        src: List = sample[self.src]
        dst = src + [EOS]
        dst = [dst[i] + dst[i + 1] for i in range(len(src))]
        sample[self.dst] = dst
        return sample


class FieldLength(NamedTransform):

    def __init__(self, src: str, dst: str = None, delta=0) -> None:
        self.delta = delta
        if not dst:
            dst = f'{src}_length'
        super().__init__(src, dst)

    def __call__(self, sample: dict) -> dict:
        sample[self.dst] = len(sample[self.src]) + self.delta
        return sample


class BMESOtoIOBES(object):
    def __init__(self, field='tag') -> None:
        self.field = field

    def __call__(self, sample: dict) -> dict:
        sample[self.field] = [self.convert(y) for y in sample[self.field]]
        return sample

    @staticmethod
    def convert(y: str):
        if y.startswith('M-'):
            return 'I-'
        return y


class NormalizeToken(ConfigurableNamedTransform):

    def __init__(self, mapper: Union[str, dict], src: str, dst: str = None) -> None:
        super().__init__(src, dst)
        self.mapper = mapper
        if isinstance(mapper, str):
            mapper = get_resource(mapper)
        if isinstance(mapper, str):
            self._table = load_json(mapper)
        elif isinstance(mapper, dict):
            self._table = mapper
        else:
            raise ValueError(f'Unrecognized mapper type {mapper}')

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if self.src == self.dst:
            sample[f'{self.src}_'] = src
        if isinstance(src, str):
            src = self.convert(src)
        else:
            src = [self.convert(x) for x in src]
        sample[self.dst] = src
        return sample

    def convert(self, token) -> str:
        return self._table.get(token, token)


class PunctuationMask(ConfigurableNamedTransform):
    def __init__(self, src: str, dst: str = None) -> None:
        """Mask out all punctuations (set mask of punctuations to False)

        Args:
          src:
          dst:

        Returns:

        """
        if not dst:
            dst = f'{src}_punct_mask'
        super().__init__(src, dst)

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            dst = not ispunct(src)
        else:
            dst = [not ispunct(x) for x in src]
        sample[self.dst] = dst
        return sample


class NormalizeCharacter(NormalizeToken):
    def convert(self, token) -> str:
        return ''.join([NormalizeToken.convert(self, c) for c in token])
