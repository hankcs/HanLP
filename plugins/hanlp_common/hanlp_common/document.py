# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 04:16
import json
import re
import warnings
from typing import List, Union

from phrasetree.tree import Tree

from hanlp_common.conll import CoNLLUWord, CoNLLSentence, CoNLLSentenceList
from hanlp_common.constant import PRED
from hanlp_common.util import collapse_json, prefix_match
from hanlp_common.visualization import tree_to_list, list_to_tree, render_labeled_span, make_table


class Document(dict):
    def __init__(self, *args, **kwargs) -> None:
        """
        A dict structure holding parsed annotations.

        Args:
            *args: An iterator of key-value pairs.
            **kwargs: Arguments from ``**`` operator.
        """
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if not v:
                continue
            if k == 'con':
                if isinstance(v, Tree) or isinstance(v[0], Tree):
                    continue
                flat = isinstance(v[0], str)
                if flat:
                    v = [v]
                ls = []
                for each in v:
                    if not isinstance(each, Tree):
                        ls.append(list_to_tree(each))
                if flat:
                    ls = ls[0]
                self[k] = ls
            elif k == 'amr':
                from hanlp_common.amr import AMRGraph
                import penman
                if isinstance(v, AMRGraph) or isinstance(v[0], AMRGraph):
                    continue
                flat = isinstance(v[0][0], str)
                if flat:
                    v = [v]
                graphs = [AMRGraph(penman.Graph(triples)) for triples in v]
                if flat:
                    graphs = graphs[0]
                self[k] = graphs

    def to_json(self, ensure_ascii=False, indent=2) -> str:
        """Convert to json string.

        Args:
            ensure_ascii: ``False`` to allow for non-ascii text.
            indent: Indent per nested structure.

        Returns:
            A text representation in ``str``.

        """
        d = self.to_dict()
        text = json.dumps(d, ensure_ascii=ensure_ascii, indent=indent, default=lambda o: repr(o))
        text = collapse_json(text, 4)
        return text

    def to_dict(self):
        """Convert to a json compatible dict.

        Returns:
            A dict representation.
        """
        d = dict(self)
        for k, v in self.items():
            if not v:
                continue
            if k == 'con':
                if not isinstance(v, Tree) and not isinstance(v[0], Tree):
                    continue
                flat = isinstance(v, Tree)
                if flat:
                    v = [v]
                ls = []
                for each in v:
                    if isinstance(each, Tree):
                        ls.append(tree_to_list(each))
                if flat:
                    ls = ls[0]
                d[k] = ls
        return d

    def __str__(self) -> str:
        return self.to_json()

    def to_conll(self, tok='tok', lem='lem', pos='pos', dep='dep', sdp='sdp') -> Union[
        CoNLLSentence, List[CoNLLSentence]]:
        """
        Convert to :class:`~hanlp_common.conll.CoNLLSentence`.

        Args:
            tok (str): Field name for tok.
            lem (str): Field name for lem.
            pos (str): Filed name for upos.
            dep (str): Field name for dependency parsing.
            sdp (str): Field name for semantic dependency parsing.

        Returns:
            A :class:`~hanlp_common.conll.CoNLLSentence` representation.

        """
        tok = prefix_match(tok, self)
        lem = prefix_match(lem, self)
        pos = prefix_match(pos, self)
        dep = prefix_match(dep, self)
        sdp = prefix_match(sdp, self)
        results = CoNLLSentenceList()
        if not self[tok]:
            return results
        flat = isinstance(self[tok][0], str)
        if flat:
            d = Document((k, [v]) for k, v in self.items())
        else:
            d = self
        for sample in [dict(zip(d, t)) for t in zip(*d.values())]:
            def get(_k, _i):
                _v = sample.get(_k, None)
                if not _v:
                    return None
                return _v[_i]

            sent = CoNLLSentence()

            for i, _tok in enumerate(sample[tok]):
                _dep = get(dep, i)
                if not _dep:
                    _dep = (None, None)
                sent.append(
                    CoNLLUWord(i + 1, form=_tok, lemma=get(lem, i), upos=get(pos, i), head=_dep[0], deprel=_dep[1],
                               deps=None if not get(sdp, i) else '|'.join(f'{x[0]}:{x[1]}' for x in get(sdp, i))))
            results.append(sent)
        if flat:
            return results[0]
        return results

    def to_pretty(self, tok='tok', lem='lem', pos='pos', dep='dep', sdp='sdp', ner='ner', srl='srl', con='con',
                  show_header=True) -> Union[str, List[str]]:
        """
        Convert to a pretty text representation which can be printed to visualize linguistic structures.

        Args:
            tok: Token key.
            lem: Lemma key.
            pos: Part-of-speech key.
            dep: Dependency parse tree key.
            sdp: Semantic dependency tree/graph key. SDP visualization has not been implemented yet.
            ner: Named entity key.
            srl: Semantic role labeling key.
            con: Constituency parsing key.
            show_header: ``True`` to print a header which indicates each field with its name.

        Returns:
            A pretty string.

        """
        results = []
        tok = prefix_match(tok, self)
        pos = prefix_match(pos, self)
        ner = prefix_match(ner, self)
        conlls = self.to_conll(tok, lem, pos, dep, sdp)
        flat = isinstance(conlls, CoNLLSentence)
        if flat:
            conlls: List[CoNLLSentence] = [conlls]

        def condense(block_, extras_=None):
            text_ = make_table(block_, insert_header=False)
            text_ = [x.split('\t', 1) for x in text_.split('\n')]
            text_ = [[x[0], x[1].replace('\t', '')] for x in text_]
            if extras_:
                for r, s in zip(extras_, text_):
                    r.extend(s)
            return text_

        for i, conll in enumerate(conlls):
            conll: CoNLLSentence = conll
            tokens = [x.form for x in conll]
            length = len(conll)
            extras = [[] for j in range(length + 1)]
            if ner in self:
                ner_samples = self[ner]
                if flat:
                    ner_samples = [ner_samples]
                ner_per_sample = ner_samples[i]
                # For nested NER, use the longest span
                start_offsets = [None for i in range(length)]
                for ent, label, b, e in ner_per_sample:
                    if not start_offsets[b] or e > start_offsets[b][-1]:
                        start_offsets[b] = (ent, label, b, e)
                ner_per_sample = [y for y in start_offsets if y]
                header = ['Tok', 'NER', 'Type']
                block = [[] for _ in range(length + 1)]
                _ner = []
                _type = []
                offset = 0
                for ent, label, b, e in ner_per_sample:
                    render_labeled_span(b, e, _ner, _type, label, offset)
                    offset = e
                if offset != length:
                    _ner.extend([''] * (length - offset))
                    _type.extend([''] * (length - offset))
                if any(_type):
                    block[0].extend(header)
                    for j, (_s, _t) in enumerate(zip(_ner, _type)):
                        block[j + 1].extend((tokens[j], _s, _t))
                    text = condense(block, extras)

            if srl in self:
                srl_samples = self[srl]
                if flat:
                    srl_samples = [srl_samples]
                srl_per_sample = srl_samples[i]
                for k, pas in enumerate(srl_per_sample):
                    if not pas:
                        continue
                    block = [[] for _ in range(length + 1)]
                    header = ['Tok', 'SRL', f'PA{k + 1}']
                    _srl = []
                    _type = []
                    offset = 0
                    p_index = None
                    for _, label, b, e in pas:
                        render_labeled_span(b, e, _srl, _type, label, offset)
                        offset = e
                        if label == PRED:
                            p_index = b
                    if len(_srl) != length:
                        _srl.extend([''] * (length - offset))
                        _type.extend([''] * (length - offset))
                    if p_index is not None:
                        _srl[p_index] = '╟──►'
                        # _type[j] = 'V'
                        if len(block) != len(_srl) + 1:
                            # warnings.warn(f'Unable to visualize overlapped spans: {pas}')
                            continue
                        block[0].extend(header)
                        for j, (_s, _t) in enumerate(zip(_srl, _type)):
                            block[j + 1].extend((tokens[j], _s, _t))
                    text = condense(block, extras)
            if con in self:
                con_samples: Tree = self[con]
                if flat:
                    con_samples: List[Tree] = [con_samples]
                tree = con_samples[i]
                block = [[] for _ in range(length + 1)]
                block[0].extend(('Tok', 'PoS'))
                for j, t in enumerate(tree.pos()):
                    block[j + 1].extend(t)

                for height in range(2, tree.height() + (0 if len(tree) == 1 else 1)):
                    offset = 0
                    spans = []
                    labels = []
                    for k, subtree in enumerate(tree.subtrees(lambda x: x.height() == height)):
                        subtree: Tree = subtree
                        b, e = offset, offset + len(subtree.leaves())
                        if height >= 3:
                            b, e = subtree[0].center, subtree[-1].center + 1
                        subtree.center = b + (e - b) // 2
                        render_labeled_span(b, e, spans, labels, subtree.label(), offset, unidirectional=True)
                        offset = e
                    if len(spans) != length:
                        spans.extend([''] * (length - len(spans)))
                    if len(labels) != length:
                        labels.extend([''] * (length - len(labels)))
                    if height < 3:
                        continue
                    block[0].extend(['', f'{height}'])
                    for j, (_s, _t) in enumerate(zip(spans, labels)):
                        block[j + 1].extend((_s, _t))
                    # check short arrows and increase their length
                    for j, arrow in enumerate(spans):
                        if not arrow:
                            # -1 current tag ; -2 arrow to current tag ; -3 = prev tag ; -4 = arrow to prev tag
                            if block[j + 1][-3] or block[j + 1][-4] == '───►':
                                if height > 3:
                                    if block[j + 1][-3]:
                                        block[j + 1][-1] = block[j + 1][-3]
                                        block[j + 1][-2] = '───►'
                                    else:
                                        block[j + 1][-1] = '────'
                                        block[j + 1][-2] = '────'
                                    block[j + 1][-3] = '────'
                                    if block[j + 1][-4] == '───►':
                                        block[j + 1][-4] = '────'
                                else:
                                    block[j + 1][-1] = '────'
                                if block[j + 1][-1] == '────':
                                    block[j + 1][-2] = '────'
                                if not block[j + 1][-4]:
                                    block[j + 1][-4] = '────'
                # If the root label is shorter than the level number, extend it to the same length
                level_len = len(block[0][-1])
                for row in block[1:]:
                    if row[-1] and len(row[-1]) < level_len:
                        row[-1] = row[-1] + ' ' * (level_len - len(row[-1]))

                text = condense(block)
                # Cosmetic issues
                for row in text:
                    while '  ─' in row[1]:
                        row[1] = row[1].replace('  ─', ' ──')
                    row[1] = row[1].replace('─  │', '───┤')
                    row[1] = row[1].replace('─  ├', '───┼')
                    row[1] = re.sub(r'►(\w+)(\s+)([│├])', lambda
                        m: f'►{m.group(1)}{"─" * len(m.group(2))}{"┤" if m.group(3) == "│" else "┼"}', row[1])
                    row[1] = re.sub(r'►(─+)►', r'─\1►', row[1])
                for r, s in zip(extras, text):
                    r.extend(s)
            # warnings.warn('Unable to visualize non-projective trees.')
            if dep in self and conll.projective:
                text = conll.to_tree(extras)
                if not show_header:
                    text = text.split('\n')
                    text = '\n'.join(text[2:])
                results.append(text)
            elif any(extras):
                results.append(make_table(extras, insert_header=True))
            else:
                results.append(' '.join(['/'.join(str(f) for f in x.nonempty_fields) for x in conll]))
        if flat:
            return results[0]
        return results

    def pretty_print(self, tok='tok', lem='lem', pos='pos', dep='dep', sdp='sdp', ner='ner', srl='srl', con='con',
                     show_header=True):
        """
        Print a pretty text representation which visualizes linguistic structures.

        Args:
            tok: Token key.
            lem: Lemma key.
            pos: Part-of-speech key.
            dep: Dependency parse tree key.
            sdp: Semantic dependency tree/graph key. SDP visualization has not been implemented yet.
            ner: Named entity key.
            srl: Semantic role labeling key.
            con: Constituency parsing key.
            show_header: ``True`` to print a header which indicates each field with its name.

        """
        results = self.to_pretty(tok, lem, pos, dep, sdp, ner, srl, con, show_header)
        if isinstance(results, str):
            results = [results]
        sent_new_line = '\n\n' if any('\n' in x for x in results) else '\n'
        print(sent_new_line.join(results))

    def translate(self, lang, tok='tok', pos='pos', dep='dep', sdp='sdp', ner='ner', srl='srl'):
        """
        Translate tags for each annotation. This is an inplace operation.

        .. Attention:: Note that the translated document might not print well in terminal due to non-ASCII characters.

        Args:
            lang: Target language to be translated to.
            tok: Token key.
            pos: Part-of-speech key.
            dep: Dependency parse tree key.
            sdp: Semantic dependency tree/graph key. SDP visualization has not been implemented yet.
            ner: Named entity key.
            srl: Semantic role labeling key.

        Returns:
            The translated document.

        """
        if lang == 'zh':
            from hanlp.utils.lang.zh import localization
        else:
            raise NotImplementedError(f'No translation for {lang}. '
                                      f'Please contribute to our translation at https://github.com/hankcs/HanLP')
        flat = isinstance(self[tok][0], str)
        for task, name in zip(['pos', 'ner', 'dep', 'sdp', 'srl'], [pos, ner, dep, sdp, srl]):
            annotations = self.get(name, None)
            if not annotations:
                continue
            if flat:
                annotations = [annotations]
            translate: dict = getattr(localization, name, None)
            if not translate:
                continue
            for anno_per_sent in annotations:
                for i, v in enumerate(anno_per_sent):
                    if task == 'ner' or task == 'dep':
                        v[1] = translate.get(v[1], v[1])
                    else:
                        anno_per_sent[i] = translate.get(v, v)
        return self

    def squeeze(self):
        r"""
        Squeeze the dimension of each field into one. It's intended to convert a nested document like ``[[sent1]]``
        to ``[sent1]``. When there are multiple sentences, only the first one will be returned. Note this is not an
        inplace operation.

        Returns:
            A squeezed document with only one sentence.

        """
        sq = Document()
        for k, v in self.items():
            sq[k] = v[0] if isinstance(v, list) else v
        return sq
