# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-02-17 15:46

_PTB_HOME = 'https://github.com/KhalilMrini/LAL-Parser/archive/master.zip#data/'

PTB_TRAIN = _PTB_HOME + '02-21.10way.clean'
'''Training set of PTB without empty categories. PoS tags are automatically predicted using 10-fold 
jackknifing (:cite:`collins-koo-2005-discriminative`).'''
PTB_DEV = _PTB_HOME + '22.auto.clean'
'''Dev set of PTB without empty categories. PoS tags are automatically predicted using 10-fold 
jackknifing (:cite:`collins-koo-2005-discriminative`).'''
PTB_TEST = _PTB_HOME + '23.auto.clean'
'''Test set of PTB without empty categories. PoS tags are automatically predicted using 10-fold 
jackknifing (:cite:`collins-koo-2005-discriminative`).'''

PTB_SD330_TRAIN = _PTB_HOME + 'ptb_train_3.3.0.sd.clean'
'''Training set of PTB in Stanford Dependencies 3.3.0 format. PoS tags are automatically predicted using 10-fold 
jackknifing (:cite:`collins-koo-2005-discriminative`).'''
PTB_SD330_DEV = _PTB_HOME + 'ptb_dev_3.3.0.sd.clean'
'''Dev set of PTB in Stanford Dependencies 3.3.0 format. PoS tags are automatically predicted using 10-fold 
jackknifing (:cite:`collins-koo-2005-discriminative`).'''
PTB_SD330_TEST = _PTB_HOME + 'ptb_test_3.3.0.sd.clean'
'''Test set of PTB in Stanford Dependencies 3.3.0 format. PoS tags are automatically predicted using 10-fold 
jackknifing (:cite:`collins-koo-2005-discriminative`).'''

PTB_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}
