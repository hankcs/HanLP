/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 18:49</create-date>
 *
 * <copyright file="WordFactory.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.document.sentence.word;

/**
 * 一个很方便的工厂类，能够自动生成不同类型的词语
 * @author hankcs
 */
public class WordFactory
{
    /**
     * 根据参数字符串产生对应的词语
     * @param param
     * @return
     */
    public static IWord create(String param)
    {
        if (param == null) return null;
        if (param.startsWith("[") && !param.startsWith("[/"))
        {
            return CompoundWord.create(param);
        }
        else
        {
            return Word.create(param);
        }
    }
}
