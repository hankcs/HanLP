/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 上午2:51</create-date>
 *
 * <copyright file="CRFTagger.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.crf.crfpp.crf_learn;
import com.hankcs.hanlp.model.crf.crfpp.Encoder;
import com.hankcs.hanlp.model.crf.crfpp.ModelImpl;
import com.hankcs.hanlp.model.crf.crfpp.TaggerImpl;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.utility.IOUtility;
import com.hankcs.hanlp.model.perceptron.utility.Utility;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Date;

/**
 * @author hankcs
 */
public abstract class CRFTagger
{
    protected ModelImpl model;

    public CRFTagger()
    {
    }


    public CRFTagger(String modelPath) throws IOException
    {
        if (modelPath == null) return; // 训练模式
        model = new ModelImpl();
        if (!model.open(IOUtil.newInputStream(modelPath), 0, 0, 1.0))
        {
            throw new IOException(String.format("%s 加载失败", modelPath));
        }
    }

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
    public void train(String templFile, String trainFile, String modelFile, boolean textModelFile,
                      int maxitr, int freq, double eta, double C, int threadNum, int shrinkingSize,
                      Encoder.Algorithm algorithm) throws IOException
    {
        Encoder encoder = new Encoder();
        if (!encoder.learn(templFile, trainFile, modelFile,
                           textModelFile, maxitr, freq, eta, C, threadNum, shrinkingSize, algorithm))
        {
            throw new IOException("fail to learn model");
        }
    }

    public void train(String trainCorpusPath, String modelPath) throws IOException
    {
        crf_learn.Option option = new crf_learn.Option();
        train(trainCorpusPath, modelPath, option.textmodel, option.maxiter, option.freq, option.eta, option.cost,
              option.thread, option.shrinking_size, Encoder.Algorithm.fromString(option.algorithm));
    }

    public void train(String trainFile, String modelFile, boolean textModelFile,
                      int maxitr, int freq, double eta, double C, int threadNum, int shrinkingSize,
                      Encoder.Algorithm algorithm) throws IOException
    {
        String templFile = null;
        File tmpTemplate = File.createTempFile("crfpp-template-" + new Date().getTime(), ".txt");
        tmpTemplate.deleteOnExit();
        templFile = tmpTemplate.getAbsolutePath();
        String template = getDefaultFeatureTemplate();
        IOUtil.saveTxt(templFile, template);

        File tmpTrain = File.createTempFile("crfpp-train-" + new Date().getTime(), ".txt");
        tmpTrain.deleteOnExit();
        convertCorpus(trainFile, tmpTrain.getAbsolutePath());
        trainFile = tmpTrain.getAbsolutePath();
        Encoder encoder = new Encoder();
        if (!encoder.learn(templFile, trainFile, modelFile,
                           textModelFile, maxitr, freq, eta, C, threadNum, shrinkingSize, algorithm))
        {
            throw new IOException("fail to learn model");
        }
    }

    protected abstract void convertCorpus(Sentence sentence, BufferedWriter bw) throws IOException;

    protected TaggerImpl createTagger()
    {
        TaggerImpl tagger = new TaggerImpl(TaggerImpl.Mode.TEST);
        tagger.setModel(this.model);
        return tagger;
    }

    protected abstract String getDefaultFeatureTemplate();

    public void convertCorpus(String pkuPath, String tsvPath) throws IOException
    {
        final BufferedWriter bw = IOUtil.newBufferedWriter(tsvPath);
        IOUtility.loadInstance(pkuPath, new InstanceHandler()
        {
            @Override
            public boolean process(Sentence sentence)
            {
                Utility.normalize(sentence);
                try
                {
                    convertCorpus(sentence, bw);
                    bw.newLine();
                }
                catch (IOException e)
                {
                    throw new RuntimeException(e);
                }
                return false;
            }
        });
        bw.close();
    }

    /**
     * 导出特征模板
     *
     * @param templatePath
     * @throws IOException
     */
    public void dumpTemplate(String templatePath) throws IOException
    {
        BufferedWriter bw = IOUtil.newBufferedWriter(templatePath);
        String template = getTemplate();
        bw.write(template);
        bw.close();
    }

    /**
     * 获取特征模板
     *
     * @return
     */
    public String getTemplate()
    {
        String template = getDefaultFeatureTemplate();
        if (model != null && model.getTemplate() != null)
        {
            template = model.getTemplate();
        }
        return template;
    }
}
