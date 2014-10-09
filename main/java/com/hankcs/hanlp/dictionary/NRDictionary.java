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
package com.hankcs.hanlp.dictionary;


import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.tag.NR;

import java.util.AbstractMap;
import java.util.Map;

/**
* 一个好用的人名词典
* @author hankcs
*/
public class NRDictionary extends CommonDictionary<EnumItem<NR>>
{
    @Override
    protected Map.Entry<String, EnumItem<NR>> onGenerateEntry(String param)
    {
        Map.Entry<String, Map.Entry<String, Integer>[]> args = EnumItem.create(param);
        EnumItem<NR> nrEnumItem = new EnumItem<>();
        for (Map.Entry<String, Integer> e : args.getValue())
        {
            nrEnumItem.labelMap.put(NR.valueOf(e.getKey()), e.getValue());
        }
        return new AbstractMap.SimpleEntry<String, EnumItem<NR>>(args.getKey(), nrEnumItem);
    }
}
