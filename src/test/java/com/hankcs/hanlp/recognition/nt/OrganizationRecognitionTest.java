package com.hankcs.hanlp.recognition.nt;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.common.CommonStringDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.utility.LexiconUtility;
import junit.framework.TestCase;

import java.util.Map;
import java.util.Set;

public class OrganizationRecognitionTest extends TestCase
{
//    public void testSeg() throws Exception
//    {
//        HanLP.Config.enableDebug();
//        DijkstraSegment segment = new DijkstraSegment();
//        segment.enableCustomDictionary(false);
//
//        segment.enableOrganizationRecognize(true);
//        System.out.println(segment.seg("东欧的球队"));
//    }
//
//    public void testGeneratePatternJavaCode() throws Exception
//    {
//        CommonStringDictionary commonStringDictionary = new CommonStringDictionary();
//        commonStringDictionary.load("data/dictionary/organization/nt.pattern.txt");
//        StringBuilder sb = new StringBuilder();
//        Set<String> keySet = commonStringDictionary.keySet();
//        CommonStringDictionary secondDictionary = new CommonStringDictionary();
//        secondDictionary.load("data/dictionary/organization/outerNT.pattern.txt");
//        keySet.addAll(secondDictionary.keySet());
//        for (String pattern : keySet)
//        {
//            sb.append("trie.addKeyword(\"" + pattern + "\");\n");
//        }
//        IOUtil.saveTxt("data/dictionary/organization/code.txt", sb.toString());
//    }
//
//    public void testRemoveP() throws Exception
//    {
//        DictionaryMaker maker = DictionaryMaker.load(HanLP.Config.OrganizationDictionaryPath);
//        for (Map.Entry<String, Item> entry : maker.entrySet())
//        {
//            String word = entry.getKey();
//            Item item = entry.getValue();
//            CoreDictionary.Attribute attribute = LexiconUtility.getAttribute(word);
//            if (attribute == null) continue;
//            if (item.containsLabel("P") && attribute.hasNatureStartsWith("u"))
//            {
//                System.out.println(item + "\t" + attribute);
//                item.removeLabel("P");
//            }
//        }
//        maker.saveTxtTo(HanLP.Config.OrganizationDictionaryPath);
//    }
}