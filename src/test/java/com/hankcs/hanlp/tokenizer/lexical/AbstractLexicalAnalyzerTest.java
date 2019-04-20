package com.hankcs.hanlp.tokenizer.lexical;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer;
import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.io.IOException;
import java.util.List;

public class AbstractLexicalAnalyzerTest extends TestCase
{
    public void testSegment() throws Exception
    {
        String[] testCase = new String[]{
            "北川景子参演了林诣彬导演的《速度与激情3》",
            "林志玲亮相网友:确定不是波多野结衣？",
            "龟山千广和近藤公园在龟山公园里喝酒赏花",
        };
        Segment segment = HanLP.newSegment("crf").enableJapaneseNameRecognize(true);
        for (String sentence : testCase)
        {
            List<Term> termList = segment.seg(sentence);
            System.out.println(termList);
        }
    }

    public void testCustomDictionary() throws Exception
    {
        LexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer();
        String text = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰";
        System.out.println(analyzer.segment(text));
        CustomDictionary.add("攻城狮");
        System.out.println(analyzer.segment(text));
    }

    public void testOverwriteTag() throws IOException
    {
        CRFLexicalAnalyzer analyzer = new CRFLexicalAnalyzer();
        String text = "强行修改词性";
        System.out.println(analyzer.seg(text));
        CustomDictionary.add("修改", "自定义词性");
        System.out.println(analyzer.seg(text));
    }
}