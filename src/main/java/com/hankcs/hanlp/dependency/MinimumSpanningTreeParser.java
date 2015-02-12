/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/26 11:35</create-date>
 *
 * <copyright file="MinimumSpanningTreeParser.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dependency.common.Edge;
import com.hankcs.hanlp.dependency.common.Node;
import com.hankcs.hanlp.dependency.common.State;
import com.hankcs.hanlp.seg.common.Term;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * @author hankcs
 */
public abstract class MinimumSpanningTreeParser extends AbstractDependencyParser
{
    @Override
    public CoNLLSentence parse(List<Term> termList)
    {
        if (termList == null || termList.size() == 0) return null;
        termList.add(0, new Term("##核心##", Nature.begin));
        Node[] nodeArray = new Node[termList.size()];
        Iterator<Term> iterator = termList.iterator();
        for (int i = 0; i < nodeArray.length; ++i)
        {
            nodeArray[i] = new Node(iterator.next(), i);
        }
        Edge[][] edges = new Edge[nodeArray.length][nodeArray.length];
        for (int i = 0; i < edges.length; ++i)
        {
            for (int j = 0; j < edges[i].length; ++j)
            {
                if (i != j)
                {
                    edges[j][i] = makeEdge(nodeArray, i, j);
                }
            }
        }
        // 最小生成树Prim算法
        int max_v = nodeArray.length * (nodeArray.length - 1);
        float[] mincost = new float[max_v];
        Arrays.fill(mincost, Float.MAX_VALUE / 3);
        boolean[] used = new boolean[max_v];
        Arrays.fill(used, false);
        used[0] = true;
        PriorityQueue<State> que = new PriorityQueue<State>();
        // 找虚根的唯一孩子
        float minCostToRoot = Float.MAX_VALUE;
        Edge firstEdge = null;
        Edge[] edgeResult = new Edge[termList.size() - 1];
        for (Edge edge : edges[0])
        {
            if (edge == null) continue;
            if (minCostToRoot > edge.cost)
            {
                firstEdge = edge;
                minCostToRoot = edge.cost;
            }
        }
        if (firstEdge == null) return null;
        que.add(new State(minCostToRoot, firstEdge.from, firstEdge));
        while (!que.isEmpty())
        {
            State p = que.poll();
            int v = p.id;
            if (used[v] || p.cost > mincost[v]) continue;
            used[v] = true;
            if (p.edge != null)
            {
//                System.out.println(p.edge.from + " " + p.edge.to + p.edge.label);
                edgeResult[p.edge.from - 1] = p.edge;
            }
            for (Edge e : edges[v])
            {
                if (e == null) continue;
                if (mincost[e.from] > e.cost)
                {
                    mincost[e.from] = e.cost;
                    que.add(new State(mincost[e.from], e.from, e));
                }
            }
        }
        CoNLLWord[] wordArray = new CoNLLWord[termList.size() - 1];
        for (int i = 0; i < wordArray.length; ++i)
        {
            wordArray[i] = new CoNLLWord(i + 1, nodeArray[i + 1].word, nodeArray[i + 1].label);
            wordArray[i].DEPREL = edgeResult[i].label;
        }
        for (int i = 0; i < edgeResult.length; ++i)
        {
            int index = edgeResult[i].to - 1;
            if (index < 0)
            {
                wordArray[i].HEAD = CoNLLWord.ROOT;
                continue;
            }
            wordArray[i].HEAD = wordArray[index];
        }
        return new CoNLLSentence(wordArray);
    }

    protected abstract Edge makeEdge(Node[] nodeArray, int from, int to);
}
