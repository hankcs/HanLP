# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-22 13:16
from hanlp_common.constant import HANLP_URL

OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH = HANLP_URL + 'mtl/open_tok_pos_ner_srl_dep_sdp_con_electra_small_20201223_035557.zip'
"Electra small version of joint tok, pos, ner, srl, dep, sdp and con model trained on open-source Chinese corpus."
OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH = HANLP_URL + 'mtl/open_tok_pos_ner_srl_dep_sdp_con_electra_base_20201223_201906.zip'
"Electra base version of joint tok, pos, ner, srl, dep, sdp and con model trained on open-source Chinese corpus."
CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH = HANLP_URL + 'mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_small_20210111_124159.zip'
"Electra small version of joint tok, pos, ner, srl, dep, sdp and con model trained on private Chinese corpus."
CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH = HANLP_URL + 'mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_base_20210111_124519.zip'
"Electra base version of joint tok, pos, ner, srl, dep, sdp and con model trained on private Chinese corpus."

UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MT5_SMALL = HANLP_URL + 'mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_mt5_small_20210228_123458.zip'
'mt5 small version of joint tok, pos, lem, fea, ner, srl, dep, sdp and con model trained on UD and OntoNotes5 corpus.'
UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE = HANLP_URL + 'mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlm_base_20210114_005825.zip'
'XLM-R base version of joint tok, pos, lem, fea, ner, srl, dep, sdp and con model trained on UD and OntoNotes5 corpus.'

NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA = HANLP_URL + 'mtl/npcmj_ud_kyoto_tok_pos_ner_dep_con_srl_bert_base_char_ja_20210517_225654.zip'
'BERT base char encoder trained on NPCMJ/UD/Kyoto corpora with encoders including tok, pos, ner, dep, con, srl.'

# Will be filled up during runtime
ALL = {}
