/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 上午1:07</create-date>
 *
 * <copyright file="CRFSegmenter.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.crf.crfpp.Encoder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Date;
//import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;

/**
 * @author hankcs
 */
public class CRFSegmenter
{
    /**
     * 训练
     *
     * @param templFile     模板文件
     * @param trainFile     训练文件
     * @param modelFile     模型文件
     * @param textModelFile 是否输出文本形式的模型文件
     * @param maxitr        最大迭代次数
     * @param freq          特征最低频次
     * @param eta           收敛阈值
     * @param C             cost-factor
     * @param threadNum     线程数
     * @param shrinkingSize
     * @param algorithm     训练算法
     * @return
     */
    public boolean train(String templFile, String trainFile, String modelFile, boolean textModelFile,
                         int maxitr, int freq, double eta, double C, int threadNum, int shrinkingSize,
                         Encoder.Algorithm algorithm)
    {
        Encoder encoder = new Encoder();
        if (!encoder.learn(templFile, trainFile, modelFile,
                           textModelFile, maxitr, freq, eta, C, threadNum, shrinkingSize, algorithm))
        {
            System.err.println("fail to learn model");
            return false;
        }
        return true;
    }

    public boolean train(String trainFile, String modelFile, boolean textModelFile,
                         int maxitr, int freq, double eta, double C, int threadNum, int shrinkingSize,
                         Encoder.Algorithm algorithm)
    {
        String templFile = null;
        try
        {
            File tmp = File.createTempFile("crfpp-template-" + new Date().getTime(), ".txt");
            tmp.deleteOnExit();
            templFile = tmp.getAbsolutePath();
            BufferedWriter bw = IOUtil.newBufferedWriter(templFile);
            String template = "# Unigram\n" +
                "U0:%x[-1,0]\n" +
                "U1:%x[0,0]\n" +
                "U2:%x[1,0]\n" +
                "U3:%x[-1,0]%x[0,0]\n" +
                "U4:%x[0,0]%x[1,0]\n" +
                "U5:%x[-1,0]%x[1,0]\n" +
                "\n" +
                "# Bigram\n" +
                "B";
            bw.write(template);
            bw.close();

        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
        Encoder encoder = new Encoder();
        if (!encoder.learn(templFile, trainFile, modelFile,
                           textModelFile, maxitr, freq, eta, C, threadNum, shrinkingSize, algorithm))
        {
            System.err.println("fail to learn model");
            return false;
        }
        return true;
    }

}
