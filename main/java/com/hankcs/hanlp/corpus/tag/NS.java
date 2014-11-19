/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/17 16:07</create-date>
 *
 * <copyright file="NS.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.tag;

/**
 * 地名角色标签
 *
 * @author hankcs
 */
public enum NS
{
    /**
     * 地名的上文 我【来到】中关园
     */
    A,
    /**
     * 地名的下文刘家村/和/下岸村/相邻
     */
    B,
    /**
     * 中国地名的第一个字
     */
    C,
    /**
     * 中国地名的第二个字
     */
    D,
    /**
     * 中国地名的第三个字
     */
    E,
    /**
     * 其他整个的地名
     */
    G,
    /**
     * 中国地名的后缀海/淀区
     */
    H,
    /**
     * 连接词刘家村/和/下岸村/相邻
     */
    X,
    /**
     * 其它非地名成分
     */
    Z,

    /**
     * 句子的开头
     */
    S,
}
