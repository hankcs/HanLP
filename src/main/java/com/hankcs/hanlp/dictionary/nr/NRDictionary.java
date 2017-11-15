/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:39</create-date>
 *
 * <copyright file="NRDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.nr;


import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.dictionary.common.EnumItemDictionary;

/**
 * 一个好用的人名词典
 *
 * @author hankcs
 */
public class NRDictionary extends EnumItemDictionary<NR>
{

    @Override
    protected NR valueOf(String name)
    {
        return NR.valueOf(name);
    }

    @Override
    protected NR[] values()
    {
        return NR.values();
    }

    @Override
    protected EnumItem<NR> newItem()
    {
        return new EnumItem<NR>();
    }
}
