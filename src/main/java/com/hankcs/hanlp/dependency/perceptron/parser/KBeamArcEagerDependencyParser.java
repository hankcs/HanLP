/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-01-08 12:35 PM</create-date>
 *
 * <copyright file="KBeamArcEagerDependencyParser.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.perceptron.parser;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.dependency.AbstractDependencyParser;
import com.hankcs.hanlp.dependency.perceptron.accessories.Evaluator;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.KBeamArcEagerParser;
import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * 基于ArcEager转移系统以平均感知机作为分类器的柱搜索依存句法分析器
 *
 * @author hankcs
 */
public class KBeamArcEagerDependencyParser extends AbstractDependencyParser
{
    KBeamArcEagerParser parser;

    public KBeamArcEagerDependencyParser() throws IOException, ClassNotFoundException
    {
        this(HanLP.Config.PerceptronParserModelPath);
    }

    public KBeamArcEagerDependencyParser(Segment segment, KBeamArcEagerParser parser)
    {
        super(segment);
        this.parser = parser;
    }

    public KBeamArcEagerDependencyParser(KBeamArcEagerParser parser)
    {
        this.parser = parser;
    }

    public KBeamArcEagerDependencyParser(String modelPath) throws IOException, ClassNotFoundException
    {
        this(new PerceptronLexicalAnalyzer(HanLP.Config.PerceptronCWSModelPath,
                                           HanLP.Config.PerceptronPOSModelPath.replaceFirst("data.*?.bin", "data/model/perceptron/ctb/pos.bin")
        ).enableCustomDictionary(false), new KBeamArcEagerParser(modelPath));
    }

    /**
     * 训练依存句法分析器
     *
     * @param trainCorpus 训练集
     * @param devCorpus   开发集
     * @param clusterPath Brown词聚类文件
     * @param modelPath   模型储存路径
     * @throws InterruptedException
     * @throws ExecutionException
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static KBeamArcEagerDependencyParser train(String trainCorpus, String devCorpus, String clusterPath, String modelPath) throws InterruptedException, ExecutionException, IOException, ClassNotFoundException
    {
        Options options = new Options();
        options.train = true;
        options.inputFile = trainCorpus;
        options.devPath = devCorpus;
        options.clusterFile = clusterPath;
        options.modelFile = modelPath;
        Main.train(options);
        return new KBeamArcEagerDependencyParser(modelPath);
    }

    /**
     * 标准化评测
     *
     * @param testCorpus 测试语料
     * @return 包含UF、LF的数组
     * @throws IOException
     * @throws ExecutionException
     * @throws InterruptedException
     */
    public double[] evaluate(String testCorpus) throws IOException, ExecutionException, InterruptedException
    {
        Options options = parser.options;
        options.goldFile = testCorpus;
        File tmpTemplate = File.createTempFile("pred-" + new Date().getTime(), ".conll");
        tmpTemplate.deleteOnExit();
        options.predFile = tmpTemplate.getAbsolutePath();
        options.outputFile = options.predFile;
        File scoreFile = File.createTempFile("score-" + new Date().getTime(), ".txt");
        scoreFile.deleteOnExit();
        parser.parseConllFile(testCorpus, options.outputFile, options.rootFirst, options.beamWidth, true,
                              options.lowercase, 1, false, scoreFile.getAbsolutePath());
        return Evaluator.evaluate(options.goldFile, options.predFile, options.punctuations);
    }

    @Override
    public CoNLLSentence parse(List<Term> termList)
    {
        return parse(termList, 64, 1);
    }

    /**
     * 执行句法分析
     *
     * @param termList     分词结果
     * @param beamWidth    柱搜索宽度
     * @param numOfThreads 多线程数
     * @return 句法树
     */
    public CoNLLSentence parse(List<Term> termList, int beamWidth, int numOfThreads)
    {
        String[] words = new String[termList.size()];
        String[] tags = new String[termList.size()];
        int k = 0;
        for (Term term : termList)
        {
            words[k] = term.word;
            tags[k] = term.nature.toString();
            ++k;
        }

        Configuration bestParse;
        try
        {
            bestParse = parser.parse(words, tags, false, beamWidth, numOfThreads);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
        CoNLLWord[] wordArray = new CoNLLWord[termList.size()];
        for (int i = 0; i < words.length; i++)
        {
            wordArray[i] = new CoNLLWord(i + 1, words[i], tags[i]);
        }
        for (int i = 0; i < words.length; i++)
        {
            wordArray[i].DEPREL = parser.idWord(bestParse.state.getDependent(i + 1));
            int index = bestParse.state.getHead(i + 1) - 1;
            if (index < 0 || index >= wordArray.length)
            {
                wordArray[i].HEAD = CoNLLWord.ROOT;
            }
            else
            {
                wordArray[i].HEAD = wordArray[index];
            }
        }
        return new CoNLLSentence(wordArray);
    }
}
