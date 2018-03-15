/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-15 下午5:39</create-date>
 *
 * <copyright file="DemoPerceptronLexicalAnalyzer.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;

import java.io.IOException;

/**
 * 基于感知机序列标注的词法分析器，默认模型训练自1998人民日报语料1月份。欢迎在更大的语料库上训练，以得到更好的效果。
 *
 * @author hankcs
 */
public class DemoPerceptronLexicalAnalyzer
{
    public static void main(String[] args) throws IOException
    {
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer();
        System.out.println(analyzer.analyze("上海华安工业（集团）公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观"));
    }
}
