package com.hankcs.hanlp.tokenizer.lexical;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

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
}