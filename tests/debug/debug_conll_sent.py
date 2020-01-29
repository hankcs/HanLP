# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-29 16:55
from hanlp.components.parsers.conll import CoNLLSentence

conll = '''\
1	蜡烛	蜡烛	NN	NN	_	3	Poss	_	_
1	蜡烛	蜡烛	NN	NN	_	4	Pat	_	_
2	两	两	CD	CD	_	3	Quan	_	_
3	头	头	NN	NN	_	4	Loc	_	_
4	烧	烧	VV	VV	_	0	Root	_	_
'''

sent = CoNLLSentence.from_str(conll)
print(sent)
print([(x.form, x.pos) for x in sent])
