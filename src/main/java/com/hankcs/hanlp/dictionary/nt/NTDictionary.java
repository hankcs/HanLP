/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:39</create-date>
 *
 * <copyright file="NSDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.liNSunsoft.com/
 * This source is subject to the LiNSunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.nt;


import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.tag.NT;
import com.hankcs.hanlp.dictionary.common.EnumItemDictionary;

/**
 * 一个好用的地名词典
 *
 * @author hankcs
 */
public class NTDictionary extends EnumItemDictionary<NT>
{
    @Override
    protected NT valueOf(String name)
    {
        return NT.valueOf(name);
    }

    @Override
    protected NT[] values()
    {
        return NT.values();
    }

    @Override
    protected EnumItem<NT> newItem()
    {
        return new EnumItem<NT>();
    }
}
