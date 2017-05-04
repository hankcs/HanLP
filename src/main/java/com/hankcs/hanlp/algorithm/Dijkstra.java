/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 12:18</create-date>
 *
 * <copyright file="Dijkstra.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algorithm;

import com.hankcs.hanlp.seg.Dijkstra.Path.State;
import com.hankcs.hanlp.seg.common.EdgeFrom;
import com.hankcs.hanlp.seg.common.Graph;
import com.hankcs.hanlp.seg.common.Vertex;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * 最短路径
 * @author hankcs
 */
public class Dijkstra
{
    public static List<Vertex> compute(Graph graph)
    {
        List<Vertex> resultList = new LinkedList<Vertex>();
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
}
