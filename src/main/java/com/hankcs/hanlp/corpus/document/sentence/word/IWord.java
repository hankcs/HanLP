/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 17:43</create-date>
 *
 * <copyright file="IWord.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.document.sentence.word;

import java.io.Serializable;

/**
 * 词语接口
 * @author hankcs
 */
public interface IWord extends Serializable
{
    String getValue();
    String getLabel();
    void setLabel(String label);
    void setValue(String value);
}
