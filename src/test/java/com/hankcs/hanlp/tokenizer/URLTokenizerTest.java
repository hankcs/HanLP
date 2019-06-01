package com.hankcs.hanlp.tokenizer;

import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.util.List;

public class URLTokenizerTest extends TestCase
{
    public void testSegment()
    {
        String text = "随便写点啥吧？abNfxbGRIAUQfGGgvesskbrhEfvCdOHyxfWBq";
        List<Term> terms = URLTokenizer.segment(text);
        System.out.println(terms);
    }
}