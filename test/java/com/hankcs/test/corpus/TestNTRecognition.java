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
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.common.CommonStringDictionary;
import com.hankcs.hanlp.seg.Dijkstra.Segment;
import junit.framework.TestCase;

import java.util.Set;

/**
 * @author hankcs
 */
public class TestNTRecognition extends TestCase
{
    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        Segment segment = new Segment();
        segment.enableCustomDictionary(false);

        segment.enableOrganizationRecognize(true);
        System.out.println(segment.seg("济南杨铭宇餐饮管理有限公司是由杨先生创办的餐饮企业"));
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
}
