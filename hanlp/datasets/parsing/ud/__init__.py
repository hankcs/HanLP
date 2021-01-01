# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-07 21:45
import os
import shutil

from hanlp.components.parsers.ud.udify_util import get_ud_treebank_files
from hanlp.utils.io_util import get_resource
from hanlp.utils.log_util import flash


def concat_treebanks(home, version):
    ud_home = get_resource(home)
    treebanks = get_ud_treebank_files(ud_home)
    output_dir = os.path.abspath(os.path.join(ud_home, os.path.pardir, os.path.pardir, f'ud-multilingual-v{version}'))
    if os.path.isdir(output_dir):
        return output_dir
    os.makedirs(output_dir)
    train, dev, test = list(zip(*[treebanks[k] for k in treebanks]))

    for treebank, name in zip([train, dev, test], ["train.conllu", "dev.conllu", "test.conllu"]):
        flash(f'Concatenating {len(train)} treebanks into {name} [blink][yellow]...[/yellow][/blink]')
        with open(os.path.join(output_dir, name), 'w') as write:
            for t in treebank:
                if not t:
                    continue
                with open(t, 'r') as read:
                    shutil.copyfileobj(read, write)
        flash('')
    return output_dir
