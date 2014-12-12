/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 21:18</create-date>
 *
 * <copyright file="TestCRF.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.model;

import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.model.crf.FeatureTemplate;
import com.hankcs.hanlp.model.crf.CRFModel;
import com.hankcs.hanlp.model.crf.Table;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.FileOutputStream;

/**
 * @author hankcs
 */
public class TestCRF extends TestCase
{
    public void testTemplate() throws Exception
    {
        FeatureTemplate featureTemplate = FeatureTemplate.create("U05:%x[-2,0]/%x[-1,0]/%x[0,0]");
        Table table = new Table();
        table.v = new String[][]{
                {"那", "S"},
                {"音", "B"},
                {"韵", "E"},};
        char[] parameter = featureTemplate.generateParameter(table, 0);
        System.out.println(parameter);
    }

    public void testTestLoadTemplate() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream("data/test/out.bin"));
        FeatureTemplate featureTemplate = FeatureTemplate.create("U05:%x[-2,0]/%x[-1,0]/%x[0,0]");
        featureTemplate.save(out);
        featureTemplate = new FeatureTemplate();
        featureTemplate.load(ByteArray.createByteArray("data/test/out.bin"));
        System.out.println(featureTemplate);
    }

    public void testLoadFromTxt() throws Exception
    {
        CRFModel model = CRFModel.loadTxt("D:\\Tools\\CRF++-0.58\\example\\seg_cn\\model.txt");
        Table table = new Table();
        table.v = new String[][]{
                {"商", "?"},
                {"品", "?"},
                {"和", "?"},
                {"服", "?"},
                {"务", "?"},
        };
        model.tag(table);
        System.out.println(table);
    }

    public void testLoadModelWhichHasNoB() throws Exception
    {
        CRFModel model = CRFModel.loadTxt("D:\\Tools\\CRF++-0.58\\example\\dependency\\model.txt");
        System.out.println(model);
    }

    public void testSegment() throws Exception
    {
//        HanLP.Config.enableDebug();
        CRFSegment segment = new CRFSegment();
//        segment.enableSpeechTag(true);
        System.out.println(segment.seg("你看过穆赫兰道吗"));
    }
}
