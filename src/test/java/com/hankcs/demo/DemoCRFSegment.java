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
 * CRF分词（目前效果还不满意，正在训练新模型，持续改进中）
 * @author hankcs
 */
public class DemoCRFSegment
{
    public static void main(String[] args)
    {
        HanLP.Config.enableDebug();
        Segment segment = new CRFSegment();
        segment.enablePartOfSpeechTagging(true);
        List<Term> termList = segment.seg("你看过穆赫兰道吗");
        System.out.println(termList);
        for (Term term : termList)
        {
            if (term.nature == null)
            {
                System.out.println("识别到新词：" + term.word);
            }
        }
    }
}
