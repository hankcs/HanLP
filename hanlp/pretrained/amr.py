# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-25 11:47
from hanlp_common.constant import HANLP_URL

AMR3_SEQ2SEQ_BART_LARGE = HANLP_URL + 'amr/amr3_seq2seq_bart_large_83.30_20220125_114450.zip'
'''A seq2seq (:cite:`bevilacqua-etal-2021-one`) BART (:cite:`lewis-etal-2020-bart`) large parser trained on Abstract 
Meaning Representation 3.0 (:cite:`knight2014abstract`). Its performance is

 =================== ========= ========= ========= 
  Metric              P         R         F1       
 =================== ========= ========= ========= 
  Smatch              84.00     82.60     83.30    
  Unlabeled           86.40     84.90     85.70    
  No WSD              84.50     83.10     83.80    
  Non_sense_frames    91.90     91.30     91.60    
  Wikification        81.70     80.80     81.20    
  Named Ent.          89.20     87.00     88.10    
  Negations           71.70     70.90     71.30    
  IgnoreVars          73.80     73.10     73.50    
  Concepts            90.70     89.60     90.10    
  Frames              88.50     87.90     88.20    
  Reentrancies        70.40     71.80     71.10    
  SRL                 79.00     79.60     79.30    
 =================== ========= ========= ========= 
    
Note this parser does NOT perform wikification.
'''

AMR3_GRAPH_PRETRAIN_PARSER = HANLP_URL + 'amr/amr3_graph_pretrain_parser_20221207_153759.zip'
'''A seq2seq (:cite:`bevilacqua-etal-2021-one`) BART (:cite:`lewis-etal-2020-bart`) large parser trained on Abstract 
Meaning Representation 3.0 (:cite:`knight2014abstract`) with graph pre-training (:cite:`bai-etal-2022-graph`). 
Its performance is ``84.3`` according to their official repository. Using ``amr-evaluation-enhanced``, the performance is
slightly lower:

 =================== ========= ========= ========= 
  Metric              P         R         F1       
 =================== ========= ========= ========= 
  Smatch             84.4       83.6        84.0       
  Unlabeled          86.7       85.8        86.2       
  No WSD             84.9       84.1        84.5       
  Non_sense_frames   91.8       91.6        91.7       
  Wikification       83.6       81.7        82.6       
  Named Ent.         89.3       87.4        88.4       
  Negations          71.6       72.2        71.9       
  IgnoreVars         74.6       74.2        74.4       
  Concepts           90.7       90.0        90.3       
  Frames             88.8       88.5        88.7       
  Reentrancies       72.1       72.9        72.5       
  SRL                80.1       80.7        80.4      
 =================== ========= ========= ========= 
    
Note this parser does NOT perform wikification.
'''

MRP2020_AMR_ENG_ZHO_XLM_BASE = 'http://download.hanlp.com/amr/extra/amr-eng-zho-xlm-roberta-base_20220412_223756.zip'
'''A wrapper for the Permutation-invariant Semantic Parser (:cite:`samuel-straka-2020-ufal`) trained on MRP2020 English 
and Chinese AMR corpus. It was ranked the top in the MRP2020 competition, while this release is a base version. 
See the original paper for the detailed performance. Note this model requires tokens and lemmas (for English) to be 
provided as inputs. 
'''

MRP2020_AMR_ZHO_MENGZI_BASE = 'http://download.hanlp.com/amr/extra/amr-zho-mengzi-base_20220415_101941.zip'
'''A Chinese Permutation-invariant Semantic Parser (:cite:`samuel-straka-2020-ufal`) trained on MRP2020  
Chinese AMR corpus using Mengzi BERT base (:cite:`zhang2021mengzi`). Its performance on dev set is 
``{amr-zho [tops F1: 85.43%][anchors F1: 93.41%][labels F1: 87.68%][properties F1: 82.02%][edges F1: 73.17%]
[attributes F1: 0.00%][all F1: 84.11%]}``. Test set performance is unknown since the test set is not released to the 
public. 
'''

# Will be filled up during runtime
ALL = {}
