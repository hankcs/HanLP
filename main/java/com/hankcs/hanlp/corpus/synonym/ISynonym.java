/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 13:05</create-date>
 *
 * <copyright file="ISynonym.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.synonym;

/**
 * 同义词接口
 * @author hankcs
 */
public interface ISynonym
{
    /**
     * 获取原本的词语
     * @return
     */
    String getRealWord();

    /**
     * 获取ID
     * @return
     */
    long getId();

    /**
     * 获取字符类型的ID
     * @return
     */
    String getIdString();
}
