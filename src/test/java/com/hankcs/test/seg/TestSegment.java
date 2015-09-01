/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/18 16:23</create-date>
 *
 * <copyright file="TestSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.dictionary.other.CharType;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import com.hankcs.hanlp.seg.Other.CommonAhoCorasickSegmentUtil;
import com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;
import com.hankcs.hanlp.seg.common.ResultTerm;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.common.wrapper.SegmentWrapper;
import com.hankcs.hanlp.tokenizer.*;
import junit.framework.TestCase;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class TestSegment extends TestCase
{
    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        Segment segment = new DijkstraSegment().enableCustomDictionary(false);
        System.out.println(segment.seg(
                "基隆市长"
        ));
    }

    public void testViterbi() throws Exception
    {
//        HanLP.Config.enableDebug(true);
        HanLP.Config.ShowTermNature = false;
        Segment segment = new DijkstraSegment();
        System.out.println(segment.seg(
                "上外日本文化经济学院的陆晚霞教授正在教授泛读课程"
        ));
    }

    public void testNotional() throws Exception
    {
        System.out.println(NotionalTokenizer.segment("算法可以宽泛的分为三类"));
    }

    public void testNGram() throws Exception
    {
        System.out.println(CoreBiGramTableDictionary.getBiFrequency("牺", "牲"));
    }

    public void testShortest() throws Exception
    {
        HanLP.Config.enableDebug();
        Segment segment = new ViterbiSegment().enableAllNamedEntityRecognize(true);
        System.out.println(segment.seg("把市场经济奉行的等价交换原则引入党的生活和国家机关政务活动中"));
    }

    public void testIndexSeg() throws Exception
    {
        System.out.println(IndexTokenizer.segment("中科院预测科学研究中心学术委员会"));
    }

    public void testWrapper() throws Exception
    {
        SegmentWrapper wrapper = new SegmentWrapper(new BufferedReader(new StringReader("中科院预测科学研究中心学术委员会\nhaha")), StandardTokenizer.SEGMENT);
        Term fullTerm;
        while ((fullTerm = wrapper.next()) != null)
        {
            System.out.println(fullTerm);
        }
    }

    public void testSpeechTagging() throws Exception
    {
        HanLP.Config.enableDebug();
        String text = "教授正在教授自然语言处理课程";
        DijkstraSegment segment = new DijkstraSegment();

        System.out.println("未标注：" + segment.seg(text));
        segment.enablePartOfSpeechTagging(true);
        System.out.println("标注后：" + segment.seg(text));
    }

    public void testFactory() throws Exception
    {
        Segment segment = HanLP.newSegment();
    }

    public void testCustomDictionary() throws Exception
    {
        DijkstraSegment segment = new DijkstraSegment();
        System.out.println(segment.seg("你在一汽马自达汽车销售有限公司上班吧"));
    }

    public void testNT() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableOrganizationRecognize(true);
        System.out.println(segment.seg("我在上海林原科技有限公司兼职工作"));
    }

    public void testACSegment() throws Exception
    {
        Segment segment = new DoubleArrayTrieSegment();
        segment.enablePartOfSpeechTagging(true);
        System.out.println(segment.seg("江西鄱阳湖干枯，中国最大淡水湖变成大草原"));
    }

    public void testIssue2() throws Exception
    {
//        HanLP.Config.enableDebug();
        String text = "BENQphone";
        System.out.println(HanLP.segment(text));
        CustomDictionary.insert("BENQ");
        System.out.println(HanLP.segment(text));
    }

    public void testIssue3() throws Exception
    {
        assertEquals(CharType.CT_DELIMITER, CharType.get('*'));;
        System.out.println(HanLP.segment("300g*2"));
        System.out.println(HanLP.segment("３００ｇ＊２"));
        System.out.println(HanLP.segment("鱼300克*2/组"));
    }

    public void testQuickAtomSegment() throws Exception
    {
        String text = "你好1234abc Good一二三四3.14";
//        System.out.println(Segment.quickAtomSegment(text.toCharArray(), 0, text.length()));
    }

    public void testJP() throws Exception
    {
        String text = "明天8.9你好abc对了";
        Segment segment = new ViterbiSegment().enableCustomDictionary(false).enableAllNamedEntityRecognize(false);
        System.out.println(segment.seg(text));
    }

    public void testSpeedOfSecondViterbi() throws Exception
    {
        String text = "王总和小丽结婚了";
        Segment segment = new ViterbiSegment().enableAllNamedEntityRecognize(false)
                .enableNameRecognize(false) // 人名识别需要二次维特比，比较慢
                .enableCustomDictionary(false)
                ;
        System.out.println(segment.seg(text));
        long start = System.currentTimeMillis();
        int pressure = 1000000;
        for (int i = 0; i < pressure; ++i)
        {
            segment.seg(text);
        }
        double costTime = (System.currentTimeMillis() - start) / (double)1000;
        System.out.printf("分词速度：%.2f字每秒", text.length() * pressure / costTime);
    }

    public void testNumberAndQuantifier() throws Exception
    {
        StandardTokenizer.SEGMENT.enableNumberQuantifierRecognize(true);
        String[] testCase = new String[]
                {
                        "十九元套餐包括什么",
                        "九千九百九十九朵玫瑰",
                        "壹佰块钱都不给我",
                        "９０１２３４５６７８只蚂蚁",
                };
        for (String sentence : testCase)
        {
            System.out.println(StandardTokenizer.segment(sentence));
        }
    }

    public void testIssue10() throws Exception
    {
        StandardTokenizer.SEGMENT.enableNumberQuantifierRecognize(true);
        IndexTokenizer.SEGMENT.enableNumberQuantifierRecognize(true);
        List termList = StandardTokenizer.segment("此帐号有欠费业务是什么");
        System.out.println(termList);
        termList = IndexTokenizer.segment("此帐号有欠费业务是什么");
        System.out.println(termList);
        termList = StandardTokenizer.segment("15307971214话费还有多少");
        System.out.println(termList);
        termList = IndexTokenizer.segment("15307971214话费还有多少");
        System.out.println(termList);
    }

    public void testMultiThreading() throws Exception
    {
        Segment segment = BasicTokenizer.SEGMENT;
        // 测个速度
        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原。";
        System.out.println(segment.seg(text));
        int pressure = 100000;
        StringBuilder sbBigText = new StringBuilder(text.length() * pressure);
        for (int i = 0; i < pressure; i++)
        {
            sbBigText.append(text);
        }
        text = sbBigText.toString();
        long start = System.currentTimeMillis();
        List<Term> termList1 = segment.seg(text);
        double costTime = (System.currentTimeMillis() - start) / (double)1000;
        System.out.printf("单线程分词速度：%.2f字每秒\n", text.length() / costTime);

        segment.enableMultithreading(4);
        start = System.currentTimeMillis();
        List<Term> termList2 = segment.seg(text);
        costTime = (System.currentTimeMillis() - start) / (double)1000;
        System.out.printf("四线程分词速度：%.2f字每秒\n", text.length() / costTime);

        assertEquals(termList1.size(), termList2.size());
        Iterator<Term> iterator1 = termList1.iterator();
        Iterator<Term> iterator2 = termList2.iterator();
        while (iterator1.hasNext())
        {
            Term term1 = iterator1.next();
            Term term2 = iterator2.next();
            assertEquals(term1.word, term2.word);
            assertEquals(term1.nature, term2.nature);
            assertEquals(term1.offset, term2.offset);
        }
    }

    public void testTryToCrashSegment() throws Exception
    {
        String text = "尝试玩坏分词器";
        Segment segment = new ViterbiSegment().enableMultithreading(100);
        System.out.println(segment.seg(text));
    }

    public void testCRFSegment() throws Exception
    {
//        HanLP.Config.enableDebug();
        HanLP.Config.ShowTermNature = false;
        Segment segment = new CRFSegment();
        System.out.println(segment.seg("尼玛不是新词，王尼玛是新词"));
        System.out.println(segment.seg("周杰伦在出品范特西之后，又出品了依然范特西"));
    }

    public void testIssue16() throws Exception
    {
        CustomDictionary.insert("爱听4g", "nz 1000");
        Segment segment = new ViterbiSegment();
        System.out.println(segment.seg("爱听4g"));
        System.out.println(segment.seg("爱听4G"));
        System.out.println(segment.seg("爱听４G"));
        System.out.println(segment.seg("爱听４Ｇ"));
        System.out.println(segment.seg("愛聽４Ｇ"));
    }

    public void testIssuse17() throws Exception
    {
        System.out.println(CharType.get('\u0000'));
        System.out.println(CharType.get(' '));
        assertEquals(CharTable.convert(' '), ' ');
        System.out.println(CharTable.convert('﹗'));
        HanLP.Config.Normalization = true;
        System.out.println(StandardTokenizer.segment("号 "));
    }

    public void testIssue22() throws Exception
    {
        CoreDictionary.Attribute attribute = CoreDictionary.get("年");
        System.out.println(attribute);
        List<Term> termList = StandardTokenizer.segment("三年");
        System.out.println(termList);
        assertEquals(attribute.nature[0], termList.get(1).nature);
        System.out.println(StandardTokenizer.segment("三元"));
        StandardTokenizer.SEGMENT.enableNumberQuantifierRecognize(true);
        System.out.println(StandardTokenizer.segment("三年"));
    }

    public void testTime() throws Exception
    {
        TraditionalChineseTokenizer.segment("认可程度");
    }

    public void testBuildASimpleSegment() throws Exception
    {
        TreeMap<String, String> dictionary = new TreeMap<String, String>();
        dictionary.put("HanLP", "名词");
        dictionary.put("特别", "副词");
        dictionary.put("方便", "形容词");
        AhoCorasickDoubleArrayTrie<String> acdat = new AhoCorasickDoubleArrayTrie<String>();
        acdat.build(dictionary);
        LinkedList<ResultTerm<String>> termList =
                CommonAhoCorasickSegmentUtil.segment("HanLP是不是特别方便？", acdat);
        System.out.println(termList);
    }

    public void testNLPSegment() throws Exception
    {
        String text = "2013年4月27日11时54分";
        NLPTokenizer.SEGMENT.enableNumberQuantifierRecognize(true);
        System.out.println(NLPTokenizer.segment(text));
    }

    public void testTraditionalSegment() throws Exception
    {
        HanLP.Config.enableDebug();
        StandardTokenizer.SEGMENT.enableAllNamedEntityRecognize(true);
        String text = "展览活动，以“高雄FUN IN中”为主题";
        System.out.println(HanLP.segment(text));
    }
}
