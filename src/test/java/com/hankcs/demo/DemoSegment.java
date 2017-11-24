/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/7 19:02</create-date>
 *
 * <copyright file="DemoSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * 标准分词
 *
 * @author hankcs
 */
public class DemoSegment
{
    public static void main(String[] args)
    {
        String[] testCase = new String[]{
                "商品和服务",
                "当下雨天地面积水分外严重",
                "结婚的和尚未结婚的确实在干扰分词啊",
                "买水果然后来世博园最后去世博会",
                "中国的首都是北京",
                "欢迎新老师生前来就餐",
                "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
                "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。",
        };
        for (String sentence : testCase)
        {
            List<Term> termList = HanLP.segment(sentence);
            System.out.println(termList);
        }
    }
}
