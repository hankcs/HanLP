/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/5 20:04</create-date>
 *
 * <copyright file="IKey.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest.scorer;

/**
 * 可以唯一代表一个句子的键，可以与其他句子区别开来
 * @author hankcs
 */
public interface ISentenceKey<T>
{
    Double similarity(T other);
}
