package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.dictionary.CustomDictionary;
import junit.framework.TestCase;

import java.util.List;

public class PerceptronSegmenterTest extends TestCase
{

    private PerceptronSegmenter segmenter;

    @Override
    public void setUp() throws Exception
    {
        segmenter = new PerceptronSegmenter();
    }

    public void testEmptyString() throws Exception
    {
        segmenter.segment("");
    }

    public void testNRF() throws Exception
    {
        String text = "他们确保了唐纳德·特朗普在总统大选中获胜。";
        List<String> wordList = segmenter.segment(text);
        assertTrue(wordList.contains("唐纳德·特朗普"));
    }

    public void testNoCustomDictionary() throws Exception
    {
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer();
        analyzer.enableCustomDictionary(false);
        CustomDictionary.insert("禁用用户词典");
        assertEquals("[禁用/v, 用户/n, 词典/n]", analyzer.seg("禁用用户词典").toString());
    }

    public void testLearnAndSeg() throws Exception
    {
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer();
        analyzer.learn("与/c 特朗普/nr 通/v 电话/n 讨论/v [太空/s 探索/vn 技术公司/n]/nt");
        assertEquals("[与/c, 特朗普/k, 通/v, 电话/n, 讨论/v, 太空探索技术公司/nt]", analyzer.seg("与特朗普通电话讨论太空探索技术公司").toString());
    }
}