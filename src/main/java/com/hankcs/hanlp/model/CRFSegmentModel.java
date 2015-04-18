/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 16:56</create-date>
 *
 * <copyright file="Index.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.crf.CRFModel;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 静态CRF分词模型
 * @author hankcs
 */
public class CRFSegmentModel
{
    public static CRFModel crfModel;
    static
    {
        logger.info("CRF分词模型正在加载 " + HanLP.Config.CRFSegmentModelPath);
        long start = System.currentTimeMillis();
        crfModel = CRFModel.loadTxt(HanLP.Config.CRFSegmentModelPath);
        if (crfModel == null)
        {
            logger.severe("CRF分词模型加载 " + HanLP.Config.CRFSegmentModelPath + " 失败，耗时 " + (System.currentTimeMillis() - start) + " ms");
            System.exit(-1);
        }
        else
            logger.info("CRF分词模型加载 " + HanLP.Config.CRFSegmentModelPath + " 成功，耗时 " + (System.currentTimeMillis() - start) + " ms");
    }
}
