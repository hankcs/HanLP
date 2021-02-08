# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-19 20:50
from typing import Union, List

from hanlp_common.structure import SerializableDict
from hanlp_common.visualization import pretty_tree_horizontal, make_table, markdown_table


class CoNLLWord(SerializableDict):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, deprel=None, phead=None,
                 pdeprel=None):
        """CoNLL (:cite:`buchholz-marsi-2006-conll`) format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf

        Args:
            id (int):
                Token counter, starting at 1 for each new sentence.
            form (str):
                Word form or punctuation symbol.
            lemma (str):
                Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
            cpos (str):
                Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
            pos (str):
                Fine-grained part-of-speech tag, where the tagset depends on the treebank.
            feats (str):
                Unordered set of syntactic and/or morphological features (depending on the particular treebank),
                or an underscore if not available.
            head (Union[int, List[int]]):
                Head of the current token, which is either a value of ID,
                or zero (’0’) if the token links to the virtual root node of the sentence.
            deprel (Union[str, List[str]]):
                Dependency relation to the HEAD.
            phead (int):
                Projective head of current token, which is either a value of ID or zero (’0’),
                or an underscore if not available.
            pdeprel (str):
                Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = sanitize_conll_int_value(id)
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = sanitize_conll_int_value(head)
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        if isinstance(self.head, list):
            return '\n'.join('\t'.join(['_' if v is None else v for v in values]) for values in [
                [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                 None if head is None else str(head), deprel, self.phead, self.pdeprel] for head, deprel in
                zip(self.head, self.deprel)
            ])
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  None if self.head is None else str(self.head), self.deprel, self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        """
        Get the values of nonempty fields as a list.
        """
        return list(f for f in
                    [self.form, self.lemma, self.cpos, self.pos, self.feats, self.head, self.deprel, self.phead,
                     self.pdeprel] if f)

    def get_pos(self):
        """
        Get the precisest pos for this word.

        Returns: ``self.pos`` or ``self.cpos``.

        """
        return self.pos or self.cpos


class CoNLLUWord(SerializableDict):
    def __init__(self, id: Union[int, str], form, lemma=None, upos=None, xpos=None, feats=None, head=None, deprel=None,
                 deps=None,
                 misc=None):
        """CoNLL-U format template, see https://universaldependencies.org/format.html

        Args:

            id (Union[int, str]):
                Token counter, starting at 1 for each new sentence.
            form (Union[str, None]):
                Word form or punctuation symbol.
            lemma (str):
                Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
            upos (str):
                Universal part-of-speech tag.
            xpos (str):
                Language-specific part-of-speech tag; underscore if not available.
            feats (str):
                List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
            head (int):
                Head of the current token, which is either a value of ID,
                or zero (’0’) if the token links to the virtual root node of the sentence.
            deprel (str):
                Dependency relation to the HEAD.
            deps (Union[List[Tuple[int, str], str]):
                Projective head of current token, which is either a value of ID or zero (’0’),
                or an underscore if not available.
            misc (str):
                Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = sanitize_conll_int_value(id)
        self.form = form
        self.upos = upos
        self.xpos = xpos
        if isinstance(head, list):
            assert deps is None, 'When head is a list, deps has to be None'
            assert isinstance(deprel, list), 'When head is a list, deprel has to be a list'
            assert len(deprel) == len(head), 'When head is a list, deprel has to match its length'
            deps = list(zip(head, deprel))
            head = None
            deprel = None
        self.head = sanitize_conll_int_value(head)
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        if deps == '_':
            deps = None
        if isinstance(deps, str):
            self.deps = []
            for pair in deps.split('|'):
                h, r = pair.split(':')
                h = int(h)
                self.deps.append((h, r))
        else:
            self.deps = deps
        self.misc = misc

    def __str__(self):
        deps = self.deps
        if not deps:
            deps = None
        else:
            deps = '|'.join(f'{h}:{r}' for h, r in deps)
        values = [str(self.id), self.form, self.lemma, self.upos, self.xpos, self.feats,
                  str(self.head) if self.head is not None else None, self.deprel, deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        """
        Get the values of nonempty fields as a list.
        """
        return list(f for f in
                    [self.form, self.lemma, self.upos, self.xpos, self.feats, self.head, self.deprel, self.deps,
                     self.misc] if f)

    def get_pos(self):
        """
        Get the precisest pos for this word.

        Returns: ``self.xpos`` or ``self.upos``

        """
        return self.xpos or self.upos


class CoNLLSentence(list):
    def __init__(self, words=None):
        """
        Create from a list of :class:`~hanlp_common.conll.CoNLLWord` or :class:`~hanlp_common.conll.CoNLLUWord`

        Args:
            words (list[Union[CoNLLWord, CoNLLUWord]]): A list of words.
        """
        super().__init__()
        if words:
            self.extend(words)

    def __str__(self):
        return '\n'.join([word.__str__() for word in self])

    @staticmethod
    def from_str(conll: str, conllu=False):
        """Build a CoNLLSentence from CoNLL-X format str

        Args:
          conll (str): CoNLL-X or CoNLL-U format string
          conllu:  ``True`` to build :class:`~hanlp_common.conll.CoNLLUWord` for each token.

        Returns:
            A :class:`~hanlp_common.conll.CoNLLSentence`.
        """
        words: List[CoNLLWord] = []
        prev_id = None
        for line in conll.strip().split('\n'):
            if line.startswith('#'):
                continue
            cells = line.split('\t')
            cells = [None if c == '_' else c for c in cells]
            if '-' in cells[0]:
                continue
            cells[0] = int(cells[0])
            cells[6] = int(cells[6])
            if cells[0] != prev_id:
                words.append(CoNLLUWord(*cells) if conllu else CoNLLWord(*cells))
            else:
                if isinstance(words[-1].head, list):
                    words[-1].head.append(cells[6])
                    words[-1].deprel.append(cells[7])
                else:
                    words[-1].head = [words[-1].head] + [cells[6]]
                    words[-1].deprel = [words[-1].deprel] + [cells[7]]
            prev_id = cells[0]
        if conllu:
            for word in words:  # type: CoNLLUWord
                if isinstance(word.head, list):
                    assert not word.deps
                    word.deps = list(zip(word.head, word.deprel))
                    word.head = None
                    word.deprel = None
        return CoNLLSentence(words)

    @staticmethod
    def from_file(path: str, conllu=False):
        """Build a CoNLLSentence from ``.conllx`` or ``.conllu`` file

        Args:
          path: Path to the file.
          conllu:  ``True`` to build :class:`~hanlp_common.conll.CoNLLUWord` for each token.

        Returns:
            A :class:`~hanlp_common.conll.CoNLLSentence`.
        """
        with open(path) as src:
            return [CoNLLSentence.from_str(x, conllu) for x in src.read().split('\n\n') if x.strip()]

    @staticmethod
    def from_dict(d: dict, conllu=False):
        """Build a CoNLLSentence from a dict.

        Args:
            d: A dict storing a list for each field, where each index corresponds to a token.
            conllu: ``True`` to build :class:`~hanlp_common.conll.CoNLLUWord` for each token.

        Returns:
            A :class:`~hanlp_common.conll.CoNLLSentence`.
        """
        if conllu:
            headings = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        else:
            headings = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
        words: List[Union[CoNLLWord, CoNLLUWord]] = []
        for cells in zip(*list(d[f] for f in headings)):
            words.append(CoNLLUWord(*cells) if conllu else CoNLLWord(*cells))
        return CoNLLSentence(words)

    def to_markdown(self, headings: Union[str, List[str]] = 'auto') -> str:
        r"""Convert into markdown string.

        Args:
            headings: ``auto`` to automatically detect the word type. When passed a list of string, they are treated as
                        headings for each field.

        Returns:
            A markdown representation of this sentence.
        """
        cells = [str(word).split('\t') for word in self]
        if headings == 'auto':
            if isinstance(self[0], CoNLLWord):
                headings = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
            else:  # conllu
                headings = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
                for each in cells:
                    # if '|' in each[8]:
                    # each[8] = f'`{each[8]}`'
                    each[8] = each[8].replace('|', '⎮')
        alignment = [('^', '>'), ('^', '<'), ('^', '<'), ('^', '<'), ('^', '<'), ('^', '<'), ('^', '>'), ('^', '<'),
                     ('^', '<'), ('^', '<')]
        text = markdown_table(headings, cells, alignment=alignment)
        return text

    def to_tree(self, extras: List[str] = None) -> str:
        """Convert into a pretty tree string which can be printed to show the tree structure.

        Args:
            extras: Extra table to be aligned to this tree.

        Returns:
            A pretty tree string along with extra table if passed any.
        """
        arrows = []
        for word in self:  # type: Union[CoNLLWord, CoNLLUWord]
            if word.head:
                arrows.append({'from': word.head - 1, 'to': word.id - 1})
        tree = pretty_tree_horizontal(arrows)
        rows = [['Dep Tree', 'Token', 'Relation']]
        has_lem = all(x.lemma for x in self)
        has_pos = all(x.get_pos() for x in self)
        if has_lem:
            rows[0].append('Lemma')
        if has_pos:
            rows[0].append('PoS')
        if extras:
            rows[0].extend(extras[0])
        for i, (word, arc) in enumerate(zip(self, tree)):
            cell_per_word = [arc]
            cell_per_word.append(word.form)
            cell_per_word.append(word.deprel)
            if has_lem:
                cell_per_word.append(word.lemma)
            if has_pos:
                cell_per_word.append(word.get_pos())
            if extras:
                cell_per_word.extend(extras[i + 1])
            rows.append(cell_per_word)
        return make_table(rows, insert_header=True)

    @property
    def projective(self):
        """
        ``True`` if this tree is projective.
        """
        return isprojective([x.head for x in self])


class CoNLLSentenceList(list):

    def __str__(self) -> str:
        return '\n\n'.join(str(x) for x in self)


def sanitize_conll_int_value(value: Union[str, int]):
    if value is None or isinstance(value, int):
        return value
    if value == '_':
        return None
    if isinstance(value, str):
        return int(value)
    return value


def isprojective(sequence):
    r"""
    Checks if a dependency tree is projective.
    This also works for partial annotation.

    Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
    which are hard to detect in the scenario of partial annotation.

    Args:
        sequence (list[int]):
            A list of head indices.

    Returns:
        ``True`` if the tree is projective, ``False`` otherwise.

    Examples:
        >>> isprojective([2, -1, 1])  # -1 denotes un-annotated cases
        False
        >>> isprojective([3, -1, 2])
        False
    """

    pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i + 1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return False
    return True
