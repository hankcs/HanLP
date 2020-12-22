/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-06 11:11 AM</create-date>
 *
 * <copyright file="TrainBigram.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch03;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;

import java.util.List;

import static com.hankcs.book.ch03.DemoCorpusLoader.MY_CWS_CORPUS_PATH;


/**
 * 《自然语言处理入门》3.3 训练
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoNgramSegment
{
    public static final String MY_MODEL_PATH = "data/test/my_cws_model";
    public static final String MSR_MODEL_PATH = MSR.MODEL_PATH + "_ngram";

    public static void main(String[] args)
    {
        trainBigram(MY_CWS_CORPUS_PATH, MY_MODEL_PATH);
        loadBigram(MY_MODEL_PATH);
        trainBigram(MSR.TRAIN_PATH, MSR_MODEL_PATH);
        loadBigram(MSR_MODEL_PATH);
    }

    /**
     * 训练bigram模型
     *
     * @param corpusPath 语料库路径
     * @param modelPath  模型保存路径
     */
    public static void trainBigram(String corpusPath, String modelPath)
    {
        List<List<IWord>> sentenceList = CorpusLoader.convert2SentenceList(corpusPath);
        for (List<IWord> sentence : sentenceList)
            for (IWord word : sentence)
                if (word.getLabel() == null) word.setLabel("n"); // 赋予每个单词一个虚拟的名词词性
        final NatureDictionaryMaker dictionaryMaker = new NatureDictionaryMaker();
        dictionaryMaker.compute(sentenceList);
        dictionaryMaker.saveTxtTo(modelPath);
    }

    public static Segment loadBigram(String modelPath)
    {
        return loadBigram(modelPath, true, true);
    }

    /**
     * 加载bigram模型
     *
     * @param modelPath 模型路径
     * @param verbose   输出调试信息
     * @param viterbi   是否创建viterbi分词器
     * @return 分词器
     */
    public static Segment loadBigram(String modelPath, boolean verbose, boolean viterbi)
    {
//        HanLP.Config.enableDebug();
        HanLP.Config.CoreDictionaryPath = modelPath + ".txt";
        HanLP.Config.BiGramDictionaryPath = modelPath + ".ngram.txt";
        CoreDictionary.reload();
        CoreBiGramTableDictionary.reload();
        // 以下部分为兼容新标注集，不感兴趣可以跳过
        HanLP.Config.CoreDictionaryTransformMatrixDictionaryPath = modelPath + ".tr.txt";
        if (!modelPath.equals(MSR_MODEL_PATH))
        {
            IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(HanLP.Config.CoreDictionaryTransformMatrixDictionaryPath);
            if (lineIterator.hasNext())
            {
                for (String tag : lineIterator.next().split(","))
                {
                    if (!tag.trim().isEmpty())
                    {
                        Nature.create(tag);
                    }
                }
            }
        }
        CoreDictionary.getTermFrequency("商品");
        CoreBiGramTableDictionary.getBiFrequency("商品", "和");
        // 兼容代码结束
        if (verbose)
        {
            HanLP.Config.ShowTermNature = false;
            System.out.println("【商品】的词频：" + CoreDictionary.getTermFrequency("商品"));
            System.out.println("【商品@和】的频次：" + CoreBiGramTableDictionary.getBiFrequency("商品", "和"));
            Segment segment = new DijkstraSegment()
                .enableAllNamedEntityRecognize(false)// 禁用命名实体识别
                .enableCustomDictionary(false); // 禁用用户词典
            System.out.println(segment.seg("商品和服务"));
//        System.out.println(segment.seg("货币和服务"));
        }
        return viterbi ? new ViterbiSegment().enableAllNamedEntityRecognize(false).enableCustomDictionary(false) :
            new DijkstraSegment().enableAllNamedEntityRecognize(false).enableCustomDictionary(false);
    }
}