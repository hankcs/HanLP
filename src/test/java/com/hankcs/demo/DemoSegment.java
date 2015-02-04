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
                "结婚的和尚未结婚的",
                "买水果然后来世博园",
                "中国的首都是北京",
                "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
        };
        Segment segment = HanLP.newSegment();
        for (String sentence : testCase)
        {
            List<Term> termList = segment.seg(sentence);
            System.out.println(termList);
        }
    }
}
