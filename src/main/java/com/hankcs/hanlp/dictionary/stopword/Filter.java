/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/28 14:12</create-date>
 *
 * <copyright file="Filter.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.stopword;

import com.hankcs.hanlp.seg.common.Term;

/**
 * 停用词词典过滤器
 * @author hankcs
 */
public interface Filter
{
    /**
     * 是否应当将这个term纳入计算
     *
     * @param term
     * @return 是否应当
     */
    boolean shouldInclude(Term term);
}
