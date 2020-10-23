# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:25
import hanlp

tokenizer = hanlp.load(hanlp.pretrained.cws.LARGE_ALBERT_BASE)
print(tokenizer('商品和服务'))
print(tokenizer(['萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。',
                 '上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。',
                 'HanLP支援臺灣正體、香港繁體，具有新詞辨識能力的中文斷詞系統']))

text = 'NLP统计模型没有加规则，聪明人知道自己加。英文、数字、自定义词典统统都是规则。'
print(tokenizer(text))

dic = {'自定义词典': 'custom_dict', '聪明人': 'smart'}


def split_by_dic(text: str):
    # We use regular expression for the sake of simplicity.
    # However, you should use some trie trees for production
    import re
    p = re.compile('(' + '|'.join(dic.keys()) + ')')
    sents, offset, words = [], 0, []
    for m in p.finditer(text):
        if offset < m.start():
            sents.append(text[offset: m.start()])
            words.append((m.group(), dic[m.group()]))
            offset = m.end()
    if offset < len(text):
        sents.append(text[offset:])
        words.append((None, None))
    flat = []
    for pred, (word, tag) in zip(tokenizer(sents), words):
        flat.extend(pred)
        if word:
            flat.append((word, tag))
    return flat


print(split_by_dic(text))
