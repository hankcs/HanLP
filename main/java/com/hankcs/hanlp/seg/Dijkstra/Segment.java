/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/29 15:14</create-date>
 *
 * <copyright file="Segment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.Dijkstra;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.recognition.nr.JapanesePersonRecogniton;
import com.hankcs.hanlp.recognition.nr.PersonRecognition;
import com.hankcs.hanlp.recognition.nr.TranslatedPersonRecognition;
import com.hankcs.hanlp.recognition.ns.PlaceRecognition;
import com.hankcs.hanlp.recognition.nt.OrganizationRecognition;
import com.hankcs.hanlp.seg.Dijkstra.Path.State;
import com.hankcs.hanlp.seg.HiddenMarkovModelSegment;
import com.hankcs.hanlp.seg.common.*;

import java.util.*;

/**
 * 最短路径分词
 * @author hankcs
 */
public class Segment extends HiddenMarkovModelSegment
{
    @Override
    public List<Term> segSentence(String sentence)
    {
        WordNet wordNetOptimum = new WordNet(sentence.toCharArray());
        WordNet wordNetAll = new WordNet(wordNetOptimum.charArray);
        ////////////////生成词网////////////////////
        GenerateWordNet(null, wordNetAll);
        ///////////////生成词图////////////////////
        Graph graph = GenerateBiGraph(wordNetAll);
        if (HanLP.Config.DEBUG)
        {
            System.out.printf("粗分词图：%s\n", graph.printByTo());
        }
        List<Vertex> vertexList = dijkstra(graph);
//        fixResultByRule(vertexList);
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
                PersonRecognition.Recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.translatedNameRecognize)
            {
                TranslatedPersonRecognition.Recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.japaneseNameRecognize)
            {
                JapanesePersonRecogniton.Recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.placeRecognize)
            {
                PlaceRecognition.Recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.organizationRecognize)
            {
                // 层叠隐马模型——生成输出作为下一级隐马输入
                graph = GenerateBiGraph(wordNetOptimum);
                vertexList = dijkstra(graph);
                wordNetOptimum.clear();
                wordNetOptimum.addAll(vertexList);
                preSize = wordNetOptimum.size();
                OrganizationRecognition.Recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (wordNetOptimum.size() != preSize)
            {
                graph = GenerateBiGraph(wordNetOptimum);
                vertexList = dijkstra(graph);
                if (HanLP.Config.DEBUG)
                {
                    System.out.printf("细分词网：\n%s\n", wordNetOptimum);
                    System.out.printf("细分词图：%s\n", graph.printByTo());
                }
            }
        }

        // 如果是索引模式则全切分
        if (config.indexMode)
        {
            return decorateResultForIndexMode(vertexList, wordNetAll);
        }

        // 是否标注词性
        if (config.speechTagging)
        {
            speechTagging(vertexList);
        }

        return convert(vertexList, config.offset);
    }

    private static List<Vertex> dijkstra(Graph graph)
    {
        List<Vertex> resultList = new LinkedList<>();
        Vertex[] vertexes = graph.getVertexes();
        List<EdgeFrom>[] edgesTo = graph.getEdgesTo();
        double[] d = new double[vertexes.length];
        Arrays.fill(d, Double.MAX_VALUE);
        d[d.length - 1] = 0;
        int[] path = new int[vertexes.length];
        Arrays.fill(path, -1);
        PriorityQueue<State> que = new PriorityQueue<State>();
        que.add(new State(0, vertexes.length - 1));
        while (!que.isEmpty())
        {
            State p = que.poll();
            if (d[p.vertex] < p.cost) continue;
            for (EdgeFrom edgeFrom : edgesTo[p.vertex])
            {
                if (d[edgeFrom.from] > d[p.vertex] + edgeFrom.weight)
                {
                    d[edgeFrom.from] = d[p.vertex] + edgeFrom.weight;
                    que.add(new State(d[edgeFrom.from], edgeFrom.from));
                    path[edgeFrom.from] = p.vertex;
                }
            }
        }
        for (int t = 0; t != -1; t = path[t])
        {
            resultList.add(vertexes[t]);
        }
        return resultList;
    }

    /**
     * 设为索引模式
     *
     * @return
     */
    public Segment enableIndexMode(boolean enable)
    {
        config.indexMode = enable;
        return this;
    }

    /**
     * 开启词性标注
     * @param enable
     * @return
     */
    public Segment enableSpeechTag(boolean enable)
    {
        config.speechTagging = enable;
        return this;
    }

    /**
     * 开启人名识别
     * @param enable
     * @return
     */
    public Segment enableNameRecognize(boolean enable)
    {
        config.nameRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 开启地名识别
     * @param enable
     * @return
     */
    public Segment enablePlaceRecognize(boolean enable)
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
    public Segment enableOrganizationRecognize(boolean enable)
    {
        config.organizationRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用用户词典
     *
     * @param enable
     */
    public Segment enableCustomDictionary(boolean enable)
    {
        config.useCustomDictionary = enable;
        return this;
    }

    /**
     * 是否启用音译人名识别
     *
     * @param enable
     */
    public Segment enableTranslatedNameRecognize(boolean enable)
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
    public Segment enableJapaneseNameRecognize(boolean enable)
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
    public Segment enableOffset(boolean enable)
    {
        config.offset = enable;
        return this;
    }

    public Segment enableAllNamedEntityRecognize(boolean enable)
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
