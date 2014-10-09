/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 17:25</create-date>
 *
 * <copyright file="Word.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.document.sentence.word;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 一个单词
 * @author hankcs
 */
public class Word implements IWord
{
    static Logger logger = LoggerFactory.getLogger(Word.class);
    /**
     * 单词的真实值，比如“程序”
     */
    public String value;
    /**
     * 单词的标签，比如“n”
     */
    public String label;

    @Override
    public String toString()
    {
        return value + '/' + label;
    }

    public Word(String value, String label)
    {
        this.value = value;
        this.label = label;
    }

    /**
     * 通过参数构造一个单词
     * @param param 比如 人民网/nz
     * @return 一个单词
     */
    public static Word create(String param)
    {
        if (param == null) return null;
        int cutIndex = param.lastIndexOf('/');
        if (cutIndex <= 0 || cutIndex == param.length() - 1)
        {
            logger.warn("使用{}创建单个单词失败", param);
            return null;
        }

        return new Word(param.substring(0, cutIndex), param.substring(cutIndex + 1));
    }

    @Override
    public String getValue()
    {
        return value;
    }

    @Override
    public String getLabel()
    {
        return label;
    }
}
