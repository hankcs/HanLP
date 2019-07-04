package com.hankcs.hanlp.model.crf;

import junit.framework.TestCase;

import java.io.IOException;

public class CRFLexicalAnalyzerTest extends TestCase
{
    public void testLoad() throws Exception
    {
        CRFLexicalAnalyzer analyzer = new CRFLexicalAnalyzer();
        String[] tests = new String[]{
            "商品和服务",
            "总统普京与特朗普通电话讨论太空探索技术公司",
            "微软公司於1975年由比爾·蓋茲和保羅·艾倫創立，18年啟動以智慧雲端、前端為導向的大改組。"
        };
//        for (String sentence : tests)
//        {
//            System.out.println(analyzer.analyze(sentence));
//            System.out.println(analyzer.seg(sentence));
//        }
    }

    public void testIssue1221() throws IOException
    {
        CRFLexicalAnalyzer analyzer = new CRFLexicalAnalyzer();
        analyzer.enableCustomDictionaryForcing(true);
        System.out.println(analyzer.seg("商品和服务"));
    }
}