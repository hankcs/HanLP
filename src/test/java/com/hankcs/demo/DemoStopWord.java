/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/6 11:11</create-date>
 *
 * <copyright file="DemoStopWordEx.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.demo;


import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.dictionary.stopword.Filter;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.BasicTokenizer;
import com.hankcs.hanlp.tokenizer.NotionalTokenizer;

import java.util.List;

import static com.hankcs.hanlp.corpus.tag.Nature.nz;

/**
 * 演示如何去除停用词
 *
 * @author hankcs
 */
public class DemoStopWord
{
    public static void main(String[] args)
    {
        String text = "小区居民有的反对喂养流浪猫，而有的居民却赞成喂养这些小宝贝";
        // 可以动态修改停用词词典
        CoreStopWordDictionary.add("居民");
        System.out.println(NotionalTokenizer.segment(text));
        CoreStopWordDictionary.remove("居民");
        System.out.println(NotionalTokenizer.segment(text));
        // 可以对任意分词器的结果执行过滤
        List<Term> termList = BasicTokenizer.segment(text);
        System.out.println(termList);
        CoreStopWordDictionary.apply(termList);
        System.out.println(termList);
        // 还可以自定义过滤逻辑
        CoreStopWordDictionary.FILTER = new Filter()
        {
            @Override
            public boolean shouldInclude(Term term)
            {
                if (term.nature == nz)
                {
                    return !CoreStopWordDictionary.contains(term.word);
                }
                return false;
            }
        };
        System.out.println(NotionalTokenizer.segment(text));
    }
}
