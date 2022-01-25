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

# Will be filled up during runtime
ALL = {}
