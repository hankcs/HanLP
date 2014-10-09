/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/17 13:25</create-date>
 *
 * <copyright file="WordResult.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort.Path;

import com.hankcs.hanlp.corpus.tag.Nature;

/**
 * 分词结果，给用户看
 * @author hankcs
 */
public class WordResult
{
    /**
     * The word
     */
    public String sWord;

    /**
     * the POS of the word
     */
    public Nature nPOS;


    public WordResult(String sWord, Nature nPOS)
    {
        this.sWord = sWord;
        this.nPOS = nPOS;
    }

    @Override
    public String toString()
    {
        return sWord + "/" + nPOS;
    }
}
