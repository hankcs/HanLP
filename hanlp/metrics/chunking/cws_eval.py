# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-06-02 22:53
# 《自然语言处理入门》2.9 准确率评测
# 配套书籍：http://nlp.hankcs.com/book.php
# 讨论答疑：https://bbs.hankcs.com/
import re


def to_region(segmentation: str) -> list:
    """将分词结果转换为区间

    Args:
      segmentation: 商品 和 服务
      segmentation: str: 

    Returns:
      0, 2), (2, 3), (3, 5)]

    """
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region


def evaluate(gold: str, pred: str, dic: dict = None) -> tuple:
    """计算P、R、F1

    Args:
      gold: 标准答案文件，比如“商品 和 服务”
      pred: 分词结果文件，比如“商品 和服 务”
      dic: 词典
      gold: str: 
      pred: str: 
      dic: dict:  (Default value = None)

    Returns:
      P, R, F1, OOV_R, IV_R)

    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    with open(gold, encoding='utf-8') as gd, open(pred, encoding='utf-8') as pd:
        for g, p in zip(gd, pd):
            A, B = set(to_region(g)), set(to_region(p))
            A_size += len(A)
            B_size += len(B)
            A_cap_B_size += len(A & B)
            text = re.sub("\\s+", "", g)
            if dic:
                for (start, end) in A:
                    word = text[start: end]
                    if word in dic:
                        IV += 1
                    else:
                        OOV += 1

                for (start, end) in A & B:
                    word = text[start: end]
                    if word in dic:
                        IV_R += 1
                    else:
                        OOV_R += 1
    p, r = safe_division(A_cap_B_size, B_size), safe_division(A_cap_B_size, A_size)
    return p, r, safe_division(2 * p * r, (p + r)), safe_division(OOV_R, OOV), safe_division(IV_R, IV)


def build_dic_from_file(path):
    dic = set()
    with open(path, encoding='utf-8') as gd:
        for g in gd:
            for word in re.compile("\\s+").split(g.strip()):
                dic.add(word)
    return dic


def safe_division(n, d):
    return n / d if d else float('nan') if n else 0.
