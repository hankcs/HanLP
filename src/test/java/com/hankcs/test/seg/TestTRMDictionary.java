/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 16:03</create-date>
 *
 * <copyright file="TestTRMDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.dictionary.TransformMatrixDictionary;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestTRMDictionary extends TestCase
{
    public void testLoad() throws Exception
    {
        TransformMatrixDictionary<NR> nrTransformMatrixDictionary = new TransformMatrixDictionary<NR>(NR.class);
        nrTransformMatrixDictionary.load("data/dictionary/person/nr.tr.txt");
        System.out.println(nrTransformMatrixDictionary.getFrequency(NR.A, NR.A));
        System.out.println(nrTransformMatrixDictionary.getFrequency("A", "A"));
        System.out.println(nrTransformMatrixDictionary.getTotalFrequency());
        System.out.println(nrTransformMatrixDictionary.getTotalFrequency(NR.Z));
        System.out.println(nrTransformMatrixDictionary.getTotalFrequency(NR.A));
    }
}
