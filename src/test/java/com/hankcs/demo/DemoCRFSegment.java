/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/10 22:02</create-date>
 *
 * <copyright file="DemoCRFSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * CRF分词(在最新训练的未压缩100MB模型下，能够取得较好的效果，可以投入生产环境)
 *
 * @author hankcs
 */
public class DemoCRFSegment
{
    public static void main(String[] args)
    {
        HanLP.Config.ShowTermNature = false;    // 关闭词性显示
        HanLP.Config.enableDebug();
        Segment segment = new CRFSegment().enableCustomDictionary(false);
        String[] sentenceArray = new String[]
                {
                        "乐视超级手机能否承载贾布斯的生态梦"
                };
        for (String sentence : sentenceArray)
        {
            List<Term> termList = segment.seg(sentence);
            System.out.println(termList);
        }
    }
}
