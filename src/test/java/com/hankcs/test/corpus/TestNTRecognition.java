/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/18 14:14</create-date>
 *
 * <copyright file="TestNTRecognition.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.common.CommonStringDictionary;
import com.hankcs.hanlp.dictionary.nt.OrganizationDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.utility.LexiconUtility;
import junit.framework.TestCase;

import java.util.Map;
import java.util.Set;

/**
 * @author hankcs
 */
public class TestNTRecognition extends TestCase
{
    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableCustomDictionary(false);

        segment.enableOrganizationRecognize(true);
        System.out.println(segment.seg("东欧的球队"));
    }

    public void testGeneratePatternJavaCode() throws Exception
    {
        CommonStringDictionary commonStringDictionary = new CommonStringDictionary();
        commonStringDictionary.load("data/dictionary/organization/nt.pattern.txt");
        StringBuilder sb = new StringBuilder();
        Set<String> keySet = commonStringDictionary.keySet();
        CommonStringDictionary secondDictionary = new CommonStringDictionary();
        secondDictionary.load("data/dictionary/organization/outerNT.pattern.txt");
        keySet.addAll(secondDictionary.keySet());
        for (String pattern : keySet)
        {
            sb.append("trie.addKeyword(\"" + pattern + "\");\n");
        }
        IOUtil.saveTxt("data/dictionary/organization/code.txt", sb.toString());
    }

    public void testRemoveP() throws Exception
    {
        DictionaryMaker maker = DictionaryMaker.load(HanLP.Config.OrganizationDictionaryPath);
        for (Map.Entry<String, Item> entry : maker.entrySet())
        {
            String word = entry.getKey();
            Item item = entry.getValue();
            CoreDictionary.Attribute attribute = LexiconUtility.getAttribute(word);
            if (attribute == null) continue;
            if (item.containsLabel("P") && attribute.hasNatureStartsWith("u"))
            {
                System.out.println(item + "\t" + attribute);
                item.removeLabel("P");
            }
        }
        maker.saveTxtTo(HanLP.Config.OrganizationDictionaryPath);
    }
}
