/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 23:04</create-date>
 *
 * <copyright file="TestCommonDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestCommonDictionary extends TestCase
{
    public void testLoad() throws Exception
    {
//        CommonDictionary<SimpleItem> dictionary = new CommonDictionary<SimpleItem>()
//        {
//            @Override
//            protected Map.Entry<String, SimpleItem> onGenerateEntry(String param)
//            {
//                String[] args = param.split(" ");
//                return new AbstractMap.SimpleEntry<>(args[0], SimpleItem.create(args));
//            }
//        };
//        dictionary.load("data/dictionary/person/nr.txt");
//        BaseSearcher searcher = dictionary.getSearcher("龚学平等领导");
//        Map.Entry<String, String> entry;
//        while ((entry = searcher.next()) != null)
//        {
//            System.out.println(entry);
//        }
    }

    public void testLoadE() throws Exception
    {
//        CommonDictionary<EnumItem<NR>> dictionary = new CommonDictionary<EnumItem<NR>>()
//        {
//            @Override
//            protected Map.Entry<String, EnumItem<NR>> onGenerateEntry(String param)
//            {
//                Map.Entry<String, Map.Entry<String, Integer>[]> args = EnumItem.create(param);
//                EnumItem<NR> nrEnumItem = new EnumItem<>();
//                for (Map.Entry<String, Integer> e : args.getValue())
//                {
//                    nrEnumItem.labelMap.put(NR.valueOf(e.getKey()), e.getValue());
//                }
//                return new AbstractMap.SimpleEntry<String, EnumItem<NR>>(args.getKey(), nrEnumItem);
//            }
//        };
//        dictionary.load("data/dictionary/person/nr.txt");
//        BaseSearcher searcher = dictionary.getSearcher("龚学平等领导");
//        Map.Entry<String, String> entry;
//        while ((entry = searcher.next()) != null)
//        {
//            System.out.println(entry);
//        }
    }
}
