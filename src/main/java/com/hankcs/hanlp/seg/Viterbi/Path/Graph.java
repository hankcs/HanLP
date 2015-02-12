/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/1/19 21:05</create-date>
 *
 * <copyright file="Grapth.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.Viterbi.Path;

import com.hankcs.hanlp.seg.common.Vertex;

import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class Graph
{
    Node nodes[][];

    public Graph(List<Vertex> vertexes[])
    {
        nodes = new Node[vertexes.length][];
        int i = 0;
        for (List<Vertex> vertexList : vertexes)
        {
            if (vertexList == null) continue;
            nodes[i] = new Node[vertexList.size()];
            int j = 0;
            for (Vertex vertex : vertexList)
            {
                nodes[i][j] = new Node(vertex);
                ++j;
            }
            ++i;
        }
    }

    public List<Vertex> viterbi()
    {
        LinkedList<Vertex> vertexList = new LinkedList<Vertex>();
        for (Node node : nodes[1])
        {
            node.updateFrom(nodes[0][0]);
        }
        for (int i = 1; i < nodes.length - 1; ++i)
        {
            Node[] nodeArray = nodes[i];
            if (nodeArray == null) continue;
            for (Node node : nodeArray)
            {
                if (node.from == null) continue;
                for (Node to : nodes[i + node.vertex.realWord.length()])
                {
                    to.updateFrom(node);
                }
            }
        }
        Node from = nodes[nodes.length - 1][0];
        while (from != null)
        {
            vertexList.addFirst(from.vertex);
            from = from.from;
        }
        return vertexList;
    }

}
