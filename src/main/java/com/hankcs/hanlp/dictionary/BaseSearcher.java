/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/10 12:10</create-date>
 *
 * <copyright file="Searcher.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;
import java.util.Map;

/**
 * 查询字典者
 * @author He Han
 */
public abstract class BaseSearcher<V>
{
    /**
     * 待分词文本的char
     */
    protected char[] c;
    /**
     * 指向当前处理字串的开始位置（前面的已经分词分完了）
     */
    protected int offset;

    protected BaseSearcher(char[] c)
    {
        this.c = c;
    }

    protected BaseSearcher(String text)
    {
        this(text.toCharArray());
    }

    /**
     * 分出下一个词
     * @return
     */
    public abstract Map.Entry<String, V> next();

    /**
     * 获取当前偏移
     * @return
     */
    public int getOffset()
    {
        return offset;
    }
}
