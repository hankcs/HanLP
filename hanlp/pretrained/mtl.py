# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-22 13:16
from hanlp_common.constant import HANLP_URL

OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH = HANLP_URL + 'mtl/open_tok_pos_ner_srl_dep_sdp_con_electra_small_20201223_035557.zip'
"Electra (:cite:`clark2020electra`) small version of joint tok, pos, ner, srl, dep, sdp and con model trained on open-source Chinese corpus."
OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH = HANLP_URL + 'mtl/open_tok_pos_ner_srl_dep_sdp_con_electra_base_20201223_201906.zip'
"Electra (:cite:`clark2020electra`) base version of joint tok, pos, ner, srl, dep, sdp and con model trained on open-source Chinese corpus."
CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH = HANLP_URL + 'mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_small_20210111_124159.zip'
"Electra (:cite:`clark2020electra`) small version of joint tok, pos, ner, srl, dep, sdp and con model trained on close-source Chinese corpus."
CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH = HANLP_URL + 'mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_base_20210111_124519.zip'
"Electra (:cite:`clark2020electra`) base version of joint tok, pos, ner, srl, dep, sdp and con model trained on close-source Chinese corpus."
CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH = HANLP_URL + 'mtl/close_tok_pos_ner_srl_dep_sdp_con_ernie_gram_base_aug_20210904_145403.zip'
"ERNIE (:cite:`xiao-etal-2021-ernie`) base version of joint tok, pos, ner, srl, dep, sdp and con model trained on close-source Chinese corpus."

UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_SMALL = HANLP_URL + 'mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlm_small_no_space_20220615_220046.zip'
'''
Distilled XLM-R (:cite:`wang-etal-2021-minilmv2`) small version of joint tok, pos, lem, fea, ner, srl, dep, sdp and con model trained on UD 2.10 and OntoNotes5 corpora.
The following 130 languages are supported: ``Afrikaans, Akkadian, Akuntsu, Albanian, Amharic, AncientGreek (to 1453), Ancient Hebrew, Apurinã, Arabic, Armenian, AssyrianNeo-Aramaic, Bambara, Basque, Beja, Belarusian, Bengali, Bhojpuri, Breton, Bulgarian, Catalan, Cebuano, Central Siberian Yupik, Chinese, Chukot, ChurchSlavic, Coptic, Croatian, Czech, Danish, Dutch, Emerillon, English, Erzya, Estonian, Faroese, Finnish, French, Galician, German, Gothic, Guajajára, Guarani, Hebrew, Hindi, Hittite, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, K\'iche\', Kangri, Karelian, Karo(Brazil), Kazakh, Khunsari, Komi-Permyak, Komi-Zyrian, Korean, Latin, Latvian, Ligurian, LiteraryChinese, Lithuanian, Livvi, LowGerman, Madi, Makuráp, Maltese, Manx, Marathi, MbyáGuaraní, Modern Greek (1453-), Moksha, Mundurukú, Nayini, Neapolitan, Nigerian Pidgin, NorthernKurdish, Northern Sami, Norwegian, OldFrench (842-ca. 1400), OldRussian, Old Turkish, Persian, Polish, Portuguese, Romanian, Russia Buriat, Russian, Sanskrit, ScottishGaelic, Serbian, SkoltSami, Slovak, Slovenian, Soi, South Levantine Arabic, Spanish, Swedish, SwedishSign Language, SwissGerman, Tagalog, Tamil, Tatar, Telugu, Thai, Tupinambá, Turkish, Uighur, Ukrainian, Umbrian, UpperSorbian, Urdu, Urubú-Kaapor, Vietnamese, Warlpiri, Welsh, Western Armenian, WesternFrisian, Wolof, Xibe, Yakut, Yoruba, YueChinese``.
Performance: ``{con UCM: 15.46% LCM: 11.63% UP: 68.87% UR: 66.84% UF: 67.84% LP: 61.19% LR: 59.39% LF: 60.28%}{ner P: 76.20% R: 77.58% F1: 76.88%}{sdp/dm UF: 91.70% LF: 90.76%}{sdp/pas UF: 95.34% LF: 93.71%}{sdp/psd UF: 91.54% LF: 79.89%}{srl [predicate P: 91.94% R: 74.09% F1: 82.06%][e2e P: 77.51% R: 55.02% F1: 64.35%]}{tok P: 94.52% R: 94.06% F1: 94.29%}{ud [lemmas Accuracy:81.69%][upos Accuracy:85.96%][deps UAS: 80.57% LAS: 71.15%][feats Accuracy:77.16%]}``.
'''
UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE = HANLP_URL + 'mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlm_base_20220608_003435.zip'
'''
XLM-R (:cite:`conneau-etal-2020-unsupervised`) base version of joint tok, pos, lem, fea, ner, srl, dep, sdp and con model trained on UD 2.10 and OntoNotes5 corpora.
The following 130 languages are supported: ``Afrikaans, Akkadian, Akuntsu, Albanian, Amharic, AncientGreek (to 1453), Ancient Hebrew, Apurinã, Arabic, Armenian, AssyrianNeo-Aramaic, Bambara, Basque, Beja, Belarusian, Bengali, Bhojpuri, Breton, Bulgarian, Catalan, Cebuano, Central Siberian Yupik, Chinese, Chukot, ChurchSlavic, Coptic, Croatian, Czech, Danish, Dutch, Emerillon, English, Erzya, Estonian, Faroese, Finnish, French, Galician, German, Gothic, Guajajára, Guarani, Hebrew, Hindi, Hittite, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, K\'iche\', Kangri, Karelian, Karo(Brazil), Kazakh, Khunsari, Komi-Permyak, Komi-Zyrian, Korean, Latin, Latvian, Ligurian, LiteraryChinese, Lithuanian, Livvi, LowGerman, Madi, Makuráp, Maltese, Manx, Marathi, MbyáGuaraní, Modern Greek (1453-), Moksha, Mundurukú, Nayini, Neapolitan, Nigerian Pidgin, NorthernKurdish, Northern Sami, Norwegian, OldFrench (842-ca. 1400), OldRussian, Old Turkish, Persian, Polish, Portuguese, Romanian, Russia Buriat, Russian, Sanskrit, ScottishGaelic, Serbian, SkoltSami, Slovak, Slovenian, Soi, South Levantine Arabic, Spanish, Swedish, SwedishSign Language, SwissGerman, Tagalog, Tamil, Tatar, Telugu, Thai, Tupinambá, Turkish, Uighur, Ukrainian, Umbrian, UpperSorbian, Urdu, Urubú-Kaapor, Vietnamese, Warlpiri, Welsh, Western Armenian, WesternFrisian, Wolof, Xibe, Yakut, Yoruba, YueChinese``.
Performance: ``{con UCM: 20.31% LCM: 16.82% UP: 77.50% UR: 76.63% UF: 77.06% LP: 71.25% LR: 70.46% LF: 70.85%}{ner P: 79.93% R: 80.76% F1: 80.34%}{sdp/dm UF: 93.71% LF: 93.00%}{sdp/pas UF: 97.63% LF: 96.37%}{sdp/psd UF: 93.08% LF: 80.95%}{srl [predicate P: 90.95% R: 84.25% F1: 87.47%][e2e P: 78.89% R: 67.32% F1: 72.65%]}{tok P: 98.50% R: 98.70% F1: 98.60%}{ud [lemmas Accuracy:85.95%][upos Accuracy:89.95%][deps UAS: 85.78% LAS: 78.51%][feats Accuracy:82.18%]}``.
'''

NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA = HANLP_URL + 'mtl/npcmj_ud_kyoto_tok_pos_ner_dep_con_srl_bert_base_char_ja_20210914_133742.zip'
'BERT (:cite:`devlin-etal-2019-bert`) base char encoder trained on NPCMJ/UD/Kyoto corpora with decoders including tok, pos, ner, dep, con, srl.'

# Will be filled up during runtime
ALL = {}
