# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-24 13:15
import hanlp

sim = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
print(sim([
    ['看图猜一电影名', '看图猜电影'],
    ['无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用'],
    ['北京到上海的动车票', '上海到北京的动车票'],
]))
