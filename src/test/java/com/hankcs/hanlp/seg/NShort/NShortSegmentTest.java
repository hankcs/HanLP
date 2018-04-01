package com.hankcs.hanlp.seg.NShort;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import junit.framework.TestCase;

import java.util.LinkedList;
import java.util.List;

public class NShortSegmentTest extends TestCase
{
    public void testParse() throws Exception
    {
        List<List<Term>> wordResults = new LinkedList<List<Term>>();
        wordResults.add(NShortSegment.parse("3-4月"));
        wordResults.add(NShortSegment.parse("3-4月份"));
        wordResults.add(NShortSegment.parse("3-4季"));
        wordResults.add(NShortSegment.parse("3-4年"));
        wordResults.add(NShortSegment.parse("3-4人"));
        wordResults.add(NShortSegment.parse("2014年"));
        wordResults.add(NShortSegment.parse("04年"));
        wordResults.add(NShortSegment.parse("12点半"));
        wordResults.add(NShortSegment.parse("1.abc"));

//        for (List<Term> result : wordResults)
//        {
//            System.out.println(result);
//        }
    }

    public void testIssue691() throws Exception
    {
//        HanLP.Config.enableDebug();
        StandardTokenizer.SEGMENT.enableCustomDictionary(false);
        Segment nShortSegment = new NShortSegment().enableCustomDictionary(false).enablePlaceRecognize(true).enableOrganizationRecognize(true);
//        System.out.println(nShortSegment.seg("今天，刘志军案的关键人物,山西女商人丁书苗在市二中院出庭受审。"));
//        System.out.println(nShortSegment.seg("今日消费5,513.58元"));
    }
}