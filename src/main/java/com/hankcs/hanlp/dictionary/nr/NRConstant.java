/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/30 23:17</create-date>
 *
 * <copyright file="NRConstant.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.nr;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.utility.Predefine;

/**
 * 人名识别中常用的一些常量
 * @author hankcs
 */
public class NRConstant
{
    /**
     * 本词典专注的词的ID
     */
    public static final int WORD_ID = CoreDictionary.getWordID(Predefine.TAG_PEOPLE);
    /**
     * 本词典专注的词的属性
     */
    public static final CoreDictionary.Attribute ATTRIBUTE = CoreDictionary.get(WORD_ID);
}
