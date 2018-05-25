/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 下午10:01</create-date>
 *
 * <copyright file="DemoCRFLexicalAnalyzer.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer;

import java.io.IOException;

/**
 * CRF词法分析器
 * @author hankcs
 */
public class DemoCRFLexicalAnalyzer
{
    public static void main(String[] args) throws IOException
    {
        CRFLexicalAnalyzer analyzer = new CRFLexicalAnalyzer();
        String[] tests = new String[]{
            "商品和服务",
            "上海华安工业（集团）公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观",
            "微软公司於1975年由比爾·蓋茲和保羅·艾倫創立，18年啟動以智慧雲端、前端為導向的大改組。" // 支持繁体中文
        };
        for (String sentence : tests)
        {
            System.out.println(analyzer.analyze(sentence));
//            System.out.println(analyzer.seg(sentence));
        }
    }
}
