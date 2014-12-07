/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/7 20:14</create-date>
 *
 * <copyright file="DemoPosTagging.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.seg.Dijkstra.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * 词性标注
 * @author hankcs
 */
public class DemoPosTagging
{
    public static void main(String[] args)
    {
        Segment segment = new Segment().enableSpeechTag(true);
        List<Term> termList = segment.seg("我的爱明明还在，爱我别走");
        System.out.println(termList);
    }
}
