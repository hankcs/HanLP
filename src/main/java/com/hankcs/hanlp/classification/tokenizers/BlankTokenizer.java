/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/10 PM5:37</create-date>
 *
 * <copyright file="BlankTokenizer.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.tokenizers;

/**
 * 使用\\s（如空白符）进行切分的分词器
 * @author hankcs
 */
public class BlankTokenizer implements ITokenizer
{
    public String[] segment(String text)
    {
        return text.split("\\s");
    }
}
