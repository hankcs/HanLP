/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/7 15:55</create-date>
 *
 * <copyright file="PhraseExactor.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.phrase;

import java.util.List;

/**
 * 从一篇文章中自动识别出最可能的短语
 * @author hankcs
 */
public interface IPhraseExtractor
{
    /**
     * 提取短语
     * @param text 文本
     * @param size 希望提取前几个短语
     * @return 短语列表
     */
    List<String> extractPhrase(String text, int size);
}
