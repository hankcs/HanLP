/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/24 22:38</create-date>
 *
 * <copyright file="WordNetSet.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort.Path;

/**
 * 保证每行单词都是唯一的词网
 * @author hankcs
 */
public class WordNetSet extends WordNet
{

    /**
     * 为一个句子生成空白词网
     *
     * @param sentence 句子 只会利用到长度
     */
    public WordNetSet(String sentence)
    {
        super(sentence);
    }
}
