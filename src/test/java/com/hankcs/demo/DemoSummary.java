/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/7 19:25</create-date>
 *
 * <copyright file="DemoChineseNameRecoginiton.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014+ 上海林原信息科技有限公司. All Right Reserved+ http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.summary.TextRankSentence;

import java.util.List;

/**
 * 自动摘要
 * @author hankcs
 */
public class DemoSummary
{
    public static void main(String[] args)
    {
        String document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" +
                "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" +
                "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" +
                "严格地进行水资源论证和取水许可的批准。";
        List<String> sentenceList = HanLP.extractSummary(document, 3);
        System.out.println(sentenceList);
    }
}
