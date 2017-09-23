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
        Segment segment = new DijkstraSegment();
        System.out.println(segment.seg(
                "我遗忘我的密码了"
        ));
    }

    public void testViterbi() throws Exception
    {
        HanLP.Config.enableDebug(true);
        CustomDictionary.add("网剧");
        Segment seg = new DijkstraSegment();
        List<Term> termList = seg.seg("优酷总裁魏明介绍了优酷2015年的内容战略，表示要以“大电影、大网剧、大综艺”为关键词");
        System.out.println(termList);
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
        CustomDictionary.insert("肯德基", "ns 1000");
        Segment segment = new ViterbiSegment();
        System.out.println(segment.seg("肯德基"));
    }

    public void testNT() throws Exception
    {
        HanLP.Config.enableDebug();
        Segment segment = new DijkstraSegment().enableOrganizationRecognize(true);
        System.out.println(segment.seg("张克智与潍坊地铁建设工程公司"));
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
        assertEquals(CharType.CT_DELIMITER, CharType.get('*'));
        System.out.println(HanLP.segment("300g*2"));
        System.out.println(HanLP.segment("３００ｇ＊２"));
        System.out.println(HanLP.segment("鱼300克*2/组"));
    }

    public void testIssue313() throws Exception
    {
        System.out.println(HanLP.segment("hello\n" + "world"));
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
                .enableCustomDictionary(false);
        System.out.println(segment.seg(text));
        long start = System.currentTimeMillis();
        int pressure = 1000000;
        for (int i = 0; i < pressure; ++i)
        {
            segment.seg(text);
        }
        double costTime = (System.currentTimeMillis() - start) / (double) 1000;
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

    public void testIssue199() throws Exception
    {
        Segment segment = new CRFSegment();
        segment.enableCustomDictionary(false);// 开启自定义词典
        segment.enablePartOfSpeechTagging(true);
        List<Term> termList = segment.seg("更多采购");
        System.out.println(termList);
        for (Term term : termList)
        {
            if (term.nature == null)
            {
                System.out.println("识别到新词：" + term.word);
            }
        }
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
        double costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("单线程分词速度：%.2f字每秒\n", text.length() / costTime);

        segment.enableMultithreading(4);
        start = System.currentTimeMillis();
        List<Term> termList2 = segment.seg(text);
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
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
        HanLP.Config.enableDebug();
//        HanLP.Config.ShowTermNature = false;
        Segment segment = new CRFSegment();
        System.out.println(segment.seg("有句谚语叫做一个萝卜一个坑儿"));
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

    public void testIssue71() throws Exception
    {
        Segment segment = HanLP.newSegment();
        segment = segment.enableAllNamedEntityRecognize(true);
        segment = segment.enableNumberQuantifierRecognize(true);
        System.out.println(segment.seg("曾幻想过，若干年后的我就是这个样子的吗"));
    }

    public void testIssue193() throws Exception
    {
        String[] testCase = new String[]{
                "以每台约200元的价格送到苹果售后维修中心换新机（苹果的保修基本是免费换新机）",
                "可能以2500~2800元的价格回收",
                "3700个益农信息社打通服务“最后一公里”",
                "一位李先生给高政留言说上周五可以帮忙献血",
                "一位浩宁达高层透露",
                "五和万科长阳天地5个普宅项目",
                "以1974点低点和5178点高点作江恩角度线",
                "纳入统计的18家京系基金公司",
                "华夏基金与嘉实基金两家京系基金公司",
                "则应从排名第八的投标人开始依次递补三名投标人"
        };
        Segment segment = HanLP.newSegment().enableOrganizationRecognize(true).enableNumberQuantifierRecognize(true);
        for (String sentence : testCase)
        {
            List<Term> termList = segment.seg(sentence);
            System.out.println(termList);
        }
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
        String text = "吵架吵到快取消結婚了";
        System.out.println(TraditionalChineseTokenizer.segment(text));
    }

    public void testIssue290() throws Exception
    {
//        HanLP.Config.enableDebug();
        String txt = "而其他肢解出去的七个贝尔公司如西南贝尔、太平洋贝尔、大西洋贝尔。";
        Segment seg_viterbi = new ViterbiSegment().enablePartOfSpeechTagging(true).enableOffset(true).enableNameRecognize(true).enablePlaceRecognize(true).enableOrganizationRecognize(true).enableNumberQuantifierRecognize(true);
        System.out.println(seg_viterbi.seg(txt));
    }

    public void testIssue343() throws Exception
    {
        CustomDictionary.insert("酷我");
        CustomDictionary.insert("酷我音乐");
        Segment segment = HanLP.newSegment().enableIndexMode(true);
        System.out.println(segment.seg("1酷我音乐2酷我音乐3酷我4酷我音乐6酷7酷我音乐"));
    }

    public void testIssue358() throws Exception
    {
        HanLP.Config.enableDebug();
        String text = "受约束，需要遵守心理学会所定的道德原则，所需要时须说明该实验与所能得到的知识的关系";

        Segment segment = StandardTokenizer.SEGMENT.enableAllNamedEntityRecognize(false).enableCustomDictionary(false)
                .enableOrganizationRecognize(true);

        System.out.println(segment.seg(text));
    }

    public void testIssue496() throws Exception
    {
        Segment segment = HanLP.newSegment().enableIndexMode(true);
        System.out.println(segment.seg("中医药"));
        System.out.println(segment.seg("中医药大学"));
    }

    public void testIssue513() throws Exception
    {
        List<Term> termList = IndexTokenizer.segment("南京市长江大桥");
        for (Term term : termList)
        {
            System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
        }
    }

    public void testIssue519() throws Exception
    {
        String[] testCase = new String[]{
            "评审委员会",
            "商标评审委员会",
            "铁道部运输局",
            "铁道部运输局营运部货运营销计划处",
        };
        for (String sentence : testCase)
        {
            System.out.println(sentence);
            List<Term> termList = IndexTokenizer.segment(sentence);
            for (Term term : termList)
            {
                System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
            }
            System.out.println();
        }
    }

    public void testIssue542() throws Exception
    {
        Segment seg = HanLP.newSegment();
        seg.enableAllNamedEntityRecognize(true);
        seg.enableNumberQuantifierRecognize(true);
        System.out.println(seg.seg("一分钟就累了"));
    }

    public void testIssue633() throws Exception
    {
        CustomDictionary.add("钱管家");
        StandardTokenizer.SEGMENT.enableCustomDictionaryForcing(true);
        System.out.println(HanLP.segment("钱管家中怎么绑定网银"));
    }
}
