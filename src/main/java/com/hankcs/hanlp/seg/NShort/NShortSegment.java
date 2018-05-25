/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/17 13:18</create-date>
 *
 * <copyright file="Segment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algorithm.Dijkstra;
import com.hankcs.hanlp.recognition.nr.JapanesePersonRecognition;
import com.hankcs.hanlp.recognition.nr.PersonRecognition;
import com.hankcs.hanlp.recognition.nr.TranslatedPersonRecognition;
import com.hankcs.hanlp.recognition.ns.PlaceRecognition;
import com.hankcs.hanlp.recognition.nt.OrganizationRecognition;
import com.hankcs.hanlp.seg.WordBasedSegment;
import com.hankcs.hanlp.seg.NShort.Path.*;
import com.hankcs.hanlp.seg.common.Graph;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.seg.common.WordNet;

import java.util.*;

/**
 * N最短分词器
 *
 * @author hankcs
 */
public class NShortSegment extends WordBasedSegment
{

    @Override
    public List<Term> segSentence(char[] sentence)
    {
        WordNet wordNetOptimum = new WordNet(sentence);
        WordNet wordNetAll = new WordNet(sentence);
//        char[] charArray = text.toCharArray();
        // 粗分
        List<List<Vertex>> coarseResult = biSegment(sentence, 2, wordNetOptimum, wordNetAll);
        boolean NERexists = false;
        for (List<Vertex> vertexList : coarseResult)
        {
            if (HanLP.Config.DEBUG)
            {
                System.out.println("粗分结果" + convert(vertexList, false));
            }
            // 实体命名识别
            if (config.ner)
            {
                wordNetOptimum.addAll(vertexList);
                int preSize = wordNetOptimum.size();
                if (config.nameRecognize)
                {
                    PersonRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
                }
                if (config.translatedNameRecognize)
                {
                    TranslatedPersonRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
                }
                if (config.japaneseNameRecognize)
                {
                    JapanesePersonRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
                }
                if (config.placeRecognize)
                {
                    PlaceRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
                }
                if (config.organizationRecognize)
                {
                    // 层叠隐马模型——生成输出作为下一级隐马输入
                    vertexList = Dijkstra.compute(generateBiGraph(wordNetOptimum));
                    wordNetOptimum.addAll(vertexList);
                    OrganizationRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
                }
                if (!NERexists && preSize != wordNetOptimum.size())
                {
                    NERexists = true;
                }
            }
        }

        List<Vertex> vertexList = coarseResult.get(0);
        if (NERexists)
        {
            Graph graph = generateBiGraph(wordNetOptimum);
            vertexList = Dijkstra.compute(graph);
            if (HanLP.Config.DEBUG)
            {
                System.out.printf("细分词网：\n%s\n", wordNetOptimum);
                System.out.printf("细分词图：%s\n", graph.printByTo());
            }
        }

        // 数字识别
        if (config.numberQuantifierRecognize)
        {
            mergeNumberQuantifier(vertexList, wordNetAll, config);
        }

        // 如果是索引模式则全切分
        if (config.indexMode > 0)
        {
            return decorateResultForIndexMode(vertexList, wordNetAll);
        }

        // 是否标注词性
        if (config.speechTagging)
        {
            speechTagging(vertexList);
        }

        if (config.useCustomDictionary)
        {
            if (config.indexMode > 0)
                combineByCustomDictionary(vertexList, wordNetAll);
            else combineByCustomDictionary(vertexList);
        }

        return convert(vertexList, config.offset);
    }

    /**
     * 二元语言模型分词
     * @param sSentence 待分词的句子
     * @param nKind     需要几个结果
     * @param wordNetOptimum
     * @param wordNetAll
     * @return 一系列粗分结果
     */
    public List<List<Vertex>> biSegment(char[] sSentence, int nKind, WordNet wordNetOptimum, WordNet wordNetAll)
    {
        List<List<Vertex>> coarseResult = new LinkedList<List<Vertex>>();
        ////////////////生成词网////////////////////
        generateWordNet(wordNetAll);
//        logger.trace("词网大小：" + wordNetAll.size());
//        logger.trace("打印词网：\n" + wordNetAll);
        ///////////////生成词图////////////////////
        Graph graph = generateBiGraph(wordNetAll);
//        logger.trace(graph.toString());
        if (HanLP.Config.DEBUG)
        {
            System.out.printf("打印词图：%s\n", graph.printByTo());
        }
        ///////////////N-最短路径////////////////////
        NShortPath nShortPath = new NShortPath(graph, nKind);
        List<int[]> spResult = nShortPath.getNPaths(nKind * 2);
        if (spResult.size() == 0)
        {
            throw new RuntimeException(nKind + "-最短路径求解失败，请检查上面的词网是否存在负圈或悬孤节点");
        }
//        logger.trace(nKind + "-最短路径");
//        for (int[] path : spResult)
//        {
//            logger.trace(Graph.parseResult(graph.parsePath(path)));
//        }
        //////////////日期、数字合并策略
        for (int[] path : spResult)
        {
            List<Vertex> vertexes = graph.parsePath(path);
            generateWord(vertexes, wordNetOptimum);
            coarseResult.add(vertexes);
        }
        return coarseResult;
    }

    /**
     * 一句话分词
     *
     * @param text
     * @return
     */
    public static List<Term> parse(String text)
    {
        return new NShortSegment().seg(text);
    }

    /**
     * 开启词性标注
     * @param enable
     * @return
     */
    public NShortSegment enablePartOfSpeechTagging(boolean enable)
    {
        config.speechTagging = enable;
        return this;
    }

    /**
     * 开启地名识别
     * @param enable
     * @return
     */
    public NShortSegment enablePlaceRecognize(boolean enable)
    {
        config.placeRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 开启机构名识别
     * @param enable
     * @return
     */
    public NShortSegment enableOrganizationRecognize(boolean enable)
    {
        config.organizationRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用音译人名识别
     *
     * @param enable
     */
    public NShortSegment enableTranslatedNameRecognize(boolean enable)
    {
        config.translatedNameRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用日本人名识别
     *
     * @param enable
     */
    public NShortSegment enableJapaneseNameRecognize(boolean enable)
    {
        config.japaneseNameRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用偏移量计算（开启后Term.offset才会被计算）
     * @param enable
     * @return
     */
    public NShortSegment enableOffset(boolean enable)
    {
        config.offset = enable;
        return this;
    }

    public NShortSegment enableAllNamedEntityRecognize(boolean enable)
    {
        config.nameRecognize = enable;
        config.japaneseNameRecognize = enable;
        config.translatedNameRecognize = enable;
        config.placeRecognize = enable;
        config.organizationRecognize = enable;
        config.updateNerConfig();
        return this;
    }

}
