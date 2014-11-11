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
import com.hankcs.hanlp.recognition.nr.PersonRecognition;
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
        WordNet wordNetOptimum = new WordNet(sentence);
        WordNet wordNetAll = new WordNet(sentence);
        ////////////////生成词网////////////////////
        GenerateWordNet(sentence, wordNetAll);
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
            System.out.println(convert(vertexList));
        }
        // 姓名识别
        if (config.nameRecognize)
        {
            wordNetOptimum.addAll(vertexList);
            int preSize = wordNetOptimum.size();
            PersonRecognition.Recognition(vertexList, wordNetOptimum, wordNetAll);
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

        return convert(vertexList);
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
     * 开启人名识别
     * @param enable
     * @return
     */
    public Segment enableNameRecognize(boolean enable)
    {
        config.nameRecognize = enable;
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
}
