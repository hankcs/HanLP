/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 20:20</create-date>
 *
 * <copyright file="NLPTokenizer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer;

import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * 可供自然语言处理用的分词器
 *
 * @author hankcs
 */
public class NLPTokenizer
{
    /**
     * 预置分词器
     */
    public static final Segment SEGMENT = new DijkstraSegment().enableNameRecognize(true).enableTranslatedNameRecognize(true)
            .enableJapaneseNameRecognize(true).enablePlaceRecognize(true).enableOrganizationRecognize(true)
            .enableSpeechTag(true);

    public static List<Term> segment(String text)
    {
        return SEGMENT.seg(text);
    }
}
