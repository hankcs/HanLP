/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/17 14:31</create-date>
 *
 * <copyright file="ISuggester.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest;

import java.util.List;

/**
 * @author hankcs
 */
public interface ISuggester
{
    void addSentence(String sentence);

    /**
     * 清空该推荐器中的所有句子
     */
    void removeAllSentences();

    /**
     * 根据一个输入的句子推荐相似的句子
     *
     * @param key
     * @param size
     * @return
     */
    List<String> suggest(String key, int size);
}
