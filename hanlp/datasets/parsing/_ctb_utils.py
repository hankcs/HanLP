# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-25 16:14
import os
import shutil
import sys
from collections import defaultdict
from os import listdir
from os.path import join, isfile
from typing import List

from phrasetree.tree import Tree

from hanlp.components.parsers.conll import read_conll
from hanlp.utils.io_util import get_resource, get_exitcode_stdout_stderr, read_tsv_as_sents, run_cmd, pushd
from hanlp.utils.log_util import cprint
from hanlp.utils.time_util import CountdownTimer


# See Shao et al., 2017
# CTB9_ACADEMIA_SPLITS = {
#     'train': '''
# 0044-0143, 0170-0270, 0400-0899,
# 1001-1017, 1019, 1021-1035, 1037-
# 1043, 1045-1059, 1062-1071, 1073-
# 1117, 1120-1131, 1133-1140, 1143-
# 1147, 1149-1151, 2000-2915, 4051-
# 4099, 4112-4180, 4198-4368, 5000-
# 5446, 6000-6560, 7000-7013
#     ''',
#     'dev': '''
# 0301-0326, 2916-3030, 4100-4106,
# 4181-4189, 4369-4390, 5447-5492,
# 6561-6630, 7013-7014
#     ''',
#     'test': '''
# 0001-0043, 0144-0169, 0271-0301,
# 0900-0931, 1018, 1020, 1036, 1044,
# 1060, 1061, 1072, 1118, 1119, 1132,
# 1141, 1142, 1148, 3031-3145, 4107-
# 4111, 4190-4197, 4391-4411, 5493-
# 5558, 6631-6700, 7015-7017
#     '''
# }
#
#
# def _make_splits(splits: Dict[str, str]):
#     total = set()
#     for part, text in list(splits.items()):
#         if not isinstance(text, str):
#             continue
#         lines = text.replace('\n', '').split()
#         cids = set()
#         for line in lines:
#             for each in line.split(','):
#                 each = each.strip()
#                 if not each:
#                     continue
#                 if '-' in each:
#                     start, end = each.split('-')
#                     start, end = map(lambda x: int(x), [start, end])
#                     cids.update(range(start, end + 1))
#                     # cids.update(map(lambda x: f'{x:04d}', range(start, end)))
#                 else:
#                     cids.add(int(each))
#         cids = set(f'{x:04d}' for x in cids)
#         assert len(cids & total) == 0, f'Overlap found in {part}'
#         splits[part] = cids
#
#     return splits
#
#
# _make_splits(CTB9_ACADEMIA_SPLITS)


def convert_to_stanford_dependency_330(src, dst, language='zh'):
    cprint(f'Converting {os.path.basename(src)} to {os.path.basename(dst)} using Stanford Parser Version 3.3.0. '
           f'It might take a while [blink][yellow]...[/yellow][/blink]')
    sp_home = 'https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip'
    sp_home = get_resource(sp_home)
    # jar_path = get_resource(f'{sp_home}#stanford-parser.jar')
    jclass = 'edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure' if language == 'zh' \
        else 'edu.stanford.nlp.trees.EnglishGrammaticalStructure'
    code, out, err = get_exitcode_stdout_stderr(
        f'java -cp {sp_home}/* {jclass} '
        f'-basic -keepPunct -conllx '
        f'-treeFile {src}')
    with open(dst, 'w') as f:
        f.write(out)
    if code:
        raise RuntimeError(f'Conversion failed with code {code} for {src}. The err message is:\n {err}\n'
                           f'Do you have java installed? Do you have enough memory?')


def clean_ctb_bracketed(ctb_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    ctb_root = join(ctb_root, 'bracketed')
    chtbs = _list_treebank_root(ctb_root)
    timer = CountdownTimer(len(chtbs))
    for f in chtbs:
        with open(join(ctb_root, f), encoding='utf-8') as src, open(join(out_root, f + '.txt'), 'w',
                                                                    encoding='utf-8') as out:
            for line in src:
                if not line.strip().startswith('<'):
                    out.write(line)
        timer.log('Cleaning up CTB [blink][yellow]...[/yellow][/blink]', erase=False)


def _list_treebank_root(ctb_root):
    chtbs = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and f.startswith('chtb')]
    return sorted(chtbs)


def list_treebank(ctb_home):
    ctb_home = get_resource(ctb_home)
    cleaned_root = join(ctb_home, 'cleaned_bracket')
    return _list_treebank_root(cleaned_root)


def load_bracketed_trees(chtbs) -> List[Tree]:
    trees = []
    for f in chtbs:
        with open(f, encoding='utf-8') as src:
            content = src.read()
            trees = [x for x in content.split('\n\n') if x.strip()]
            for tree in trees:
                tree = Tree.fromstring(tree)
                trees.append(tree)
    return trees


def split_str_to_trees(text: str):
    trees = []
    buffer = []
    for line in text.split('\n'):
        if not line.strip():
            continue
        if line.startswith('('):
            if buffer:
                trees.append('\n'.join(buffer).strip())
                buffer = []
        buffer.append(line)
    if buffer:
        trees.append('\n'.join(buffer).strip())
    return trees


def make_ctb_tasks(chtbs, out_root, part):
    for task in ['cws', 'pos', 'par', 'dep']:
        os.makedirs(join(out_root, task), exist_ok=True)
    timer = CountdownTimer(len(chtbs))
    par_path = join(out_root, 'par', f'{part}.txt')
    with open(join(out_root, 'cws', f'{part}.txt'), 'w', encoding='utf-8') as cws, \
            open(join(out_root, 'pos', f'{part}.tsv'), 'w', encoding='utf-8') as pos, \
            open(par_path, 'w', encoding='utf-8') as par:
        for f in chtbs:
            with open(f, encoding='utf-8') as src:
                content = src.read()
                trees = split_str_to_trees(content)
                for tree in trees:
                    try:
                        tree = Tree.fromstring(tree)
                    except ValueError:
                        print(tree)
                        exit(1)
                    words = []
                    for word, tag in tree.pos():
                        if tag == '-NONE-' or not tag:
                            continue
                        tag = tag.split('-')[0]
                        if tag == 'X':  # 铜_NN 30_CD ｘ_X 25_CD ｘ_X 14_CD cm_NT 1999_NT
                            tag = 'FW'
                        pos.write('{}\t{}\n'.format(word, tag))
                        words.append(word)
                    cws.write(' '.join(words))
                    par.write(tree.pformat(margin=sys.maxsize))
                    for fp in cws, pos, par:
                        fp.write('\n')
            timer.log(f'Preprocesing the [blue]{part}[/blue] set of CTB [blink][yellow]...[/yellow][/blink]',
                      erase=False)
    remove_all_ec(par_path)
    dep_path = join(out_root, 'dep', f'{part}.conllx')
    convert_to_stanford_dependency_330(par_path, dep_path)
    sents = list(read_conll(dep_path))
    with open(dep_path, 'w') as out:
        for sent in sents:
            for i, cells in enumerate(sent):
                tag = cells[3]
                tag = tag.split('-')[0]  # NT-SHORT ---> NT
                if tag == 'X':  # 铜_NN 30_CD ｘ_X 25_CD ｘ_X 14_CD cm_NT 1999_NT
                    tag = 'FW'
                cells[3] = cells[4] = tag
                out.write('\t'.join(str(x) for x in cells))
                out.write('\n')
            out.write('\n')


def reverse_splits(splits):
    cid_domain = dict()
    for domain, cids in splits.items():
        for each in cids:
            cid_domain[each] = domain
    return cid_domain


def split_chtb(chtbs: List[str], splits=None):
    train, dev, test = [], [], []
    unused = []
    for each in chtbs:
        name, domain, ext = each.split('.', 2)
        _, cid = name.split('_')
        if splits:
            if cid in splits['train']:
                bin = train
            elif cid in splits['dev']:
                bin = dev
            elif cid in splits['test']:
                bin = test
            else:
                bin = unused
                # raise IOError(f'{name} not in any splits')
        else:
            bin = train
            if name.endswith('8'):
                bin = dev
            elif name.endswith('9'):
                bin = test
        bin.append(each)
    return train, dev, test


def id_of_chtb(each: str):
    return int(each.split('.')[0].split('_')[-1])


def make_ctb(ctb_home):
    ctb_home = get_resource(ctb_home)
    cleaned_root = join(ctb_home, 'cleaned_bracket')
    if not os.path.isdir(cleaned_root):
        clean_ctb_bracketed(ctb_home, cleaned_root)
    tasks_root = join(ctb_home, 'tasks')
    if not os.path.isdir(tasks_root):
        try:
            chtbs = _list_treebank_root(cleaned_root)
            print(f'For the {len(chtbs)} files in CTB, we apply the following splits:')
            train, dev, test = split_chtb(chtbs)
            for part, name in zip([train, dev, test], ['train', 'dev', 'test']):
                print(f'{name} = {[id_of_chtb(x) for x in part]}')
            cprint('[yellow]Each file id ending with 8/9 is put into '
                   'dev/test respectively, the rest are put into train. '
                   'Our splits ensure files are evenly split across each genre, which is recommended '
                   'for production systems.[/yellow]')
            for part, name in zip([train, dev, test], ['train', 'dev', 'test']):
                make_ctb_tasks([join(cleaned_root, x) for x in part], tasks_root, name)
            cprint('Done pre-processing CTB. Enjoy your research with [blue]HanLP[/blue]!')
        except Exception as e:
            shutil.rmtree(tasks_root, ignore_errors=True)
            raise e


def load_domains(ctb_home):
    """
    Load file ids from a Chinese treebank grouped by domains.

    Args:
        ctb_home: Root path to CTB.

    Returns:
        A dict of sets, each represents a domain.
    """
    ctb_home = get_resource(ctb_home)
    ctb_root = join(ctb_home, 'bracketed')
    chtbs = _list_treebank_root(ctb_root)
    domains = defaultdict(set)
    for each in chtbs:
        name, domain = each.split('.')
        _, fid = name.split('_')
        domains[domain].add(fid)
    return domains


def ctb_pos_to_text_format(path, delimiter='_'):
    """
    Convert ctb pos tagging corpus from tsv format to text format, where each word is followed by
    its pos tag.
    Args:
        path: File to be converted.
        delimiter: Delimiter between word and tag.
    """
    path = get_resource(path)
    name, ext = os.path.splitext(path)
    with open(f'{name}.txt', 'w', encoding='utf-8') as out:
        for sent in read_tsv_as_sents(path):
            out.write(' '.join([delimiter.join(x) for x in sent]))
            out.write('\n')


def remove_all_ec(path):
    """
    Remove empty categories for all trees in this file and save them into a "noempty" file.

    Args:
        path: File path.
    """
    script = get_resource('https://file.hankcs.com/bin/remove_ec.zip')
    with pushd(script):
        run_cmd(f'java -cp elit-ddr-0.0.5-SNAPSHOT.jar:elit-sdk-0.0.5-SNAPSHOT.jar:hanlp-1.7.8.jar:'
                f'fastutil-8.1.1.jar:. demo.RemoveEmptyCategoriesTreebank {path}')
