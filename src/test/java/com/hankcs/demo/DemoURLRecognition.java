/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/3/18 AM10:17</create-date>
 *
 * <copyright file="DemoURLRecognition.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.URLTokenizer;

import java.util.List;

/**
 * 演示URL识别
 *
 * @author hankcs
 */
public class DemoURLRecognition
{
    public static void main(String[] args)
    {
        String text =
                "HanLP的项目地址是https://github.com/hankcs/HanLP，" +
                        "发布地址是https://github.com/hankcs/HanLP/releases，" +
                        "我有时候会在www.hankcs.com上面发布一些消息，" +
                        "我的微博是http://weibo.com/hankcs/，会同步推送hankcs.com的新闻。" +
                        "听说.中国域名开放申请了,但我并没有申请hankcs.中国,因为穷……";
        List<Term> termList = URLTokenizer.segment(text);
        System.out.println(termList);
        for (Term term : termList)
        {
            if (term.nature == Nature.xu)
                System.out.println(term.word);
        }
    }
}
