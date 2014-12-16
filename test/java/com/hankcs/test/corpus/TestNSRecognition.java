/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/17 14:32</create-date>
 *
 * <copyright file="TestNSRecognization.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestNSRecognition extends TestCase
{
    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableJapaneseNameRecognize(false);
        segment.enableTranslatedNameRecognize(false);
        segment.enableNameRecognize(false);
        segment.enableCustomDictionary(false);

        segment.enablePlaceRecognize(true);
        System.out.println(segment.seg("南翔向宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"));
    }
}
