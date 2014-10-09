/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/14 19:22</create-date>
 *
 * <copyright file="ChainContent.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.dynamicarray;

/**
 * @author He Han
 */
public class ChainContent
{
    public String sWord;
    public int nPOS;
    public double eWeight;

    public ChainContent()
    {
    }

    public ChainContent(double eWeight)
    {
        this.eWeight = eWeight;
    }

    public ChainContent(double eWeight, int nPos)
    {
        this.eWeight = eWeight;
        this.nPOS = nPos;
    }

    public ChainContent(double eWeight, int nPos, String sWord)
    {
        this.eWeight = eWeight;
        this.nPOS = nPos;
        this.sWord = sWord;
    }

}
