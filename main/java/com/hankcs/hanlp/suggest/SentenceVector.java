/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 20:56</create-date>
 *
 * <copyright file="Sentence.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest;

import com.hankcs.hanlp.dictionary.CommonSynonymDictionary;
import com.hankcs.hanlp.dictionary.CoreSynonymDictionary;
import com.hankcs.hanlp.seg.NShort.Segment;
import com.hankcs.hanlp.utility.CharUtility;
import com.hankcs.hanlp.utility.CharUtility;

import java.util.List;

/**
 * 推荐系统中的句子的词向量
 * @author hankcs
 */
public class SentenceVector implements Comparable<SentenceVector>
{
    /**
     * 语义向量
     */
    long[] lexemeArray;

    /**
     * 用于再次计算的词向量
     */
    List<CommonSynonymDictionary.SynonymItem> synonymItemList;

    /**
     * 独一无二的key
     */
    String key;

    public SentenceVector(String sentence, boolean withUndefined)
    {
        this(CoreSynonymDictionary.convert(Segment.parse(sentence), withUndefined));
    }

    public SentenceVector(List<CommonSynonymDictionary.SynonymItem> synonymItemList)
    {
        this.lexemeArray = CoreSynonymDictionary.getLexemeArray(synonymItemList);
        StringBuilder sbKey = new StringBuilder(lexemeArray.length * 4);
        for (long x : lexemeArray)
        {
            sbKey.append(CharUtility.long2String(x));
        }
        this.key = sbKey.toString();
        this.synonymItemList = synonymItemList;
    }

    @Override
    public int compareTo(SentenceVector o)
    {
        return key.compareTo(o.key);
    }
}
