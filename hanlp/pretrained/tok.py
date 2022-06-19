# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:12
from hanlp_common.constant import HANLP_URL

SIGHAN2005_PKU_CONVSEG = HANLP_URL + 'tok/sighan2005-pku-convseg_20200110_153722.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on sighan2005 pku dataset.'
SIGHAN2005_MSR_CONVSEG = HANLP_URL + 'tok/convseg-msr-nocrf-noembed_20200110_153524.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on sighan2005 msr dataset.'
CTB6_CONVSEG = HANLP_URL + 'tok/ctb6_convseg_nowe_nocrf_20200110_004046.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on CTB6 dataset.'
PKU_NAME_MERGED_SIX_MONTHS_CONVSEG = HANLP_URL + 'tok/pku98_6m_conv_ngram_20200110_134736.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on pku98 six months dataset with familiy name and given name merged into one unit.'
LARGE_ALBERT_BASE = HANLP_URL + 'tok/large_corpus_cws_albert_base_20211228_160926.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on the largest CWS dataset in the world.'
SIGHAN2005_PKU_BERT_BASE_ZH = HANLP_URL + 'tok/sighan2005_pku_bert_base_zh_20201231_141130.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on sighan2005 pku dataset.'
COARSE_ELECTRA_SMALL_ZH = HANLP_URL + 'tok/coarse_electra_small_20220616_012050.zip'
'Electra (:cite:`clark2020electra`) small model trained on coarse-grained CWS corpora. Its performance is ``P: 98.34% R: 98.38% F1: 98.36%`` which is ' \
'much higher than that of MTL model '
FINE_ELECTRA_SMALL_ZH = HANLP_URL + 'tok/fine_electra_small_20220615_231803.zip'
'Electra (:cite:`clark2020electra`) small model trained on fine-grained CWS corpora. Its performance is ``P: 98.14% R: 98.07% F1: 98.11%`` which is ' \
'much higher than that of MTL model '
CTB9_TOK_ELECTRA_SMALL = HANLP_URL + 'tok/ctb9_electra_small_20220215_205427.zip'
'Electra (:cite:`clark2020electra`) small model trained on CTB9. Its performance is P=97.15% R=97.36% F1=97.26% which is ' \
'much higher than that of MTL model '
CTB9_TOK_ELECTRA_BASE = 'http://download.hanlp.com/tok/extra/ctb9_tok_electra_base_20220426_111949.zip'
'Electra (:cite:`clark2020electra`) base model trained on CTB9. Its performance is ``P: 97.62% R: 97.67% F1: 97.65%`` ' \
'which is much higher than that of MTL model '
CTB9_TOK_ELECTRA_BASE_CRF = 'http://download.hanlp.com/tok/extra/ctb9_tok_electra_base_crf_20220426_161255.zip'
'Electra (:cite:`clark2020electra`) base model trained on CTB9. Its performance is ``P: 97.68% R: 97.71% F1: 97.69%`` ' \
'which is much higher than that of MTL model '
MSR_TOK_ELECTRA_BASE_CRF = 'http://download.hanlp.com/tok/extra/msra_crf_electra_base_20220507_113936.zip'
'Electra (:cite:`clark2020electra`) base model trained on MSR CWS dataset. Its performance is ``P: 98.71% R: 98.64% F1: 98.68%`` ' \
'which is much higher than that of MTL model '

UD_TOK_MMINILMV2L6 = HANLP_URL + 'tok/ud_tok_mMiniLMv2L6_no_space_mul_20220619_091824.zip'
'''
mMiniLMv2 (:cite:`wang-etal-2021-minilmv2`) L6xH384 based tokenizer trained on UD 2.10.
The following 130 languages are supported: ``Afrikaans, Akkadian, Akuntsu, Albanian, Amharic, AncientGreek (to 1453), Ancient Hebrew, Apurinã, Arabic, Armenian, AssyrianNeo-Aramaic, Bambara, Basque, Beja, Belarusian, Bengali, Bhojpuri, Breton, Bulgarian, Catalan, Cebuano, Central Siberian Yupik, Chinese, Chukot, ChurchSlavic, Coptic, Croatian, Czech, Danish, Dutch, Emerillon, English, Erzya, Estonian, Faroese, Finnish, French, Galician, German, Gothic, Guajajára, Guarani, Hebrew, Hindi, Hittite, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, K\'iche\', Kangri, Karelian, Karo(Brazil), Kazakh, Khunsari, Komi-Permyak, Komi-Zyrian, Korean, Latin, Latvian, Ligurian, LiteraryChinese, Lithuanian, Livvi, LowGerman, Madi, Makuráp, Maltese, Manx, Marathi, MbyáGuaraní, Modern Greek (1453-), Moksha, Mundurukú, Nayini, Neapolitan, Nigerian Pidgin, NorthernKurdish, Northern Sami, Norwegian, OldFrench (842-ca. 1400), OldRussian, Old Turkish, Persian, Polish, Portuguese, Romanian, Russia Buriat, Russian, Sanskrit, ScottishGaelic, Serbian, SkoltSami, Slovak, Slovenian, Soi, South Levantine Arabic, Spanish, Swedish, SwedishSign Language, SwissGerman, Tagalog, Tamil, Tatar, Telugu, Thai, Tupinambá, Turkish, Uighur, Ukrainian, Umbrian, UpperSorbian, Urdu, Urubú-Kaapor, Vietnamese, Warlpiri, Welsh, Western Armenian, WesternFrisian, Wolof, Xibe, Yakut, Yoruba, YueChinese``.
Performance: ``P: 94.99% R: 94.74% F1: 94.86%``.
'''
UD_TOK_MMINILMV2L12 = HANLP_URL + 'tok/ud_tok_mMiniLMv2L12_no_space_mul_20220619_091159.zip'
'''
mMiniLMv2 (:cite:`wang-etal-2021-minilmv2`) L12xH384 based tokenizer trained on UD 2.10.
The following 130 languages are supported: ``Afrikaans, Akkadian, Akuntsu, Albanian, Amharic, AncientGreek (to 1453), Ancient Hebrew, Apurinã, Arabic, Armenian, AssyrianNeo-Aramaic, Bambara, Basque, Beja, Belarusian, Bengali, Bhojpuri, Breton, Bulgarian, Catalan, Cebuano, Central Siberian Yupik, Chinese, Chukot, ChurchSlavic, Coptic, Croatian, Czech, Danish, Dutch, Emerillon, English, Erzya, Estonian, Faroese, Finnish, French, Galician, German, Gothic, Guajajára, Guarani, Hebrew, Hindi, Hittite, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, K\'iche\', Kangri, Karelian, Karo(Brazil), Kazakh, Khunsari, Komi-Permyak, Komi-Zyrian, Korean, Latin, Latvian, Ligurian, LiteraryChinese, Lithuanian, Livvi, LowGerman, Madi, Makuráp, Maltese, Manx, Marathi, MbyáGuaraní, Modern Greek (1453-), Moksha, Mundurukú, Nayini, Neapolitan, Nigerian Pidgin, NorthernKurdish, Northern Sami, Norwegian, OldFrench (842-ca. 1400), OldRussian, Old Turkish, Persian, Polish, Portuguese, Romanian, Russia Buriat, Russian, Sanskrit, ScottishGaelic, Serbian, SkoltSami, Slovak, Slovenian, Soi, South Levantine Arabic, Spanish, Swedish, SwedishSign Language, SwissGerman, Tagalog, Tamil, Tatar, Telugu, Thai, Tupinambá, Turkish, Uighur, Ukrainian, Umbrian, UpperSorbian, Urdu, Urubú-Kaapor, Vietnamese, Warlpiri, Welsh, Western Armenian, WesternFrisian, Wolof, Xibe, Yakut, Yoruba, YueChinese``.
Performance: ``P: 95.41% R: 95.25% F1: 95.33%``.
'''

# Will be filled up during runtime
ALL = {}
