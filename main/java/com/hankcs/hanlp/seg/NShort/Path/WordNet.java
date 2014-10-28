/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/19 21:06</create-date>
 *
 * <copyright file="Graph.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort.Path;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.MathTools;
import com.hankcs.hanlp.utility.Predefine;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.regex.Pattern;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * @author hankcs
 */
public class WordNet
{
    /**
     * 节点，每一行都是前缀词，跟图的表示方式不同
     */
    private List<Vertex> vertexes[];

    /**
     * 共有多少个节点
     */
    int size;

    /**
     * 原始句子
     */
    public String sentence;

    /**
     * 为一个句子生成空白词网
     * @param sentence 句子
     */
    public WordNet(String sentence)
    {
        this.sentence = sentence;
        vertexes = new List[sentence.length() + 2];
        for (int i = 0; i < vertexes.length; ++i)
        {
            vertexes[i] = new LinkedList<Vertex>();
        }
        vertexes[0].add(Vertex.B);
        vertexes[vertexes.length - 1].add(Vertex.E);
        size = 2;
    }

    /**
     * 添加顶点
     *
     * @param line   行号
     * @param vertex 顶点
     */
    public void add(int line, Vertex vertex)
    {
        for (Vertex oldVertex : vertexes[line])
        {
            // 保证唯一性
            if (oldVertex.realWord.length() == vertex.realWord.length()) return;
        }
        vertexes[line].add(vertex);
        ++size;
    }

    /**
     * 全自动添加顶点
     * @param vertexList
     */
    public void addAll(List<Vertex> vertexList)
    {
        int i = 0;
        for (Vertex vertex : vertexList)
        {
            add(i, vertex);
            i += vertex.realWord.length();
        }
    }

    /**
     * 获取某一行的所有节点
     * @param line 行号
     * @return 一个数组
     */
    public List<Vertex> get(int line)
    {
        return vertexes[line];
    }

    /**
     * 添加顶点，由原子分词顶点添加
     *
     * @param line
     * @param atomSegment
     */
    public void add(int line, List<AtomNode> atomSegment)
    {
        // 将原子部分存入m_segGraph
        int offset = 0;
        for (AtomNode atomNode : atomSegment)//Init the cost array
        {
            String sWord = atomNode.sWord;//init the word
            Nature nature = Nature.n;
            int dValue = Predefine.MAX_FREQUENCY;
            switch (atomNode.nPOS)
            {
                case Predefine.CT_CHINESE:
                    break;
                case Predefine.CT_INDEX:
                case Predefine.CT_NUM:
                    nature = Nature.m;
                    sWord = "未##数";
                    dValue = 0;
                    break;
                case Predefine.CT_DELIMITER:
                    nature = Nature.w;
                    break;
                case Predefine.CT_LETTER:
                    nature = Nature.nx;
                    dValue = 0;
                    sWord = "未##串";
                    break;
                case Predefine.CT_SINGLE://12021-2129-3121
                    if (Pattern.compile("^(-?\\d+)(\\.\\d+)?$").matcher(sWord).matches())//匹配浮点数
                    {
                        nature = Nature.m;
                        sWord = "未##数";
                    } else
                    {
                        nature = Nature.nx;
                        sWord = "未##串";
                    }
                    dValue = 0;
                    break;
                default:
                    break;
            }
            add(line + offset, new Vertex(sWord, atomNode.sWord, new CoreDictionary.Attribute(nature, dValue)));
            offset += atomNode.sWord.length();
        }
    }

    public int getSize()
    {
        return size;
    }

    /**
     * 获取顶点数组
     *
     * @return Vertex[] 按行优先列次之的顺序构造的顶点数组
     */
    private Vertex[] getVertexes()
    {
        Vertex[] vertexes = new Vertex[size];
        int i = 0;
        for (List<Vertex> vertexList : this.vertexes)
        {
            for (Vertex v : vertexList)
            {
                v.index = i;    // 设置id
                vertexes[i++] = v;
            }
        }

        return vertexes;
    }

    /**
     * 词网转词图
     * @return 词图
     */
    public Graph toGraph()
    {
        Graph graph = new Graph(getVertexes());

        for (int row = 0; row < vertexes.length - 1; ++row)
        {
            List<Vertex> vertexListFrom = vertexes[row];
            for (Vertex from : vertexListFrom)
            {
                assert from.realWord.length() > 0 : "空节点会导致死循环！";
                int toIndex = row + from.realWord.length();
                for (Vertex to : vertexes[toIndex])
                {
                    graph.connect(from.index, to.index, MathTools.calculateWeight(from, to));
                }
            }
        }
        return graph;
    }

    @Override
    public String toString()
    {
//        return "Graph{" +
//                "vertexes=" + Arrays.toString(vertexes) +
//                '}';
        StringBuilder sb = new StringBuilder();
        int line = 0;
        for (List<Vertex> vertexList : vertexes)
        {
            sb.append(String.valueOf(line++) + ':' + vertexList.toString()).append("\n");
        }
        return sb.toString();
    }

    /**
     * 将连续的ns节点合并为一个
     */
    public void mergeContinuousNsIntoOne()
    {
        for (int row = 0; row < vertexes.length - 1; ++row)
        {
            List<Vertex> vertexListFrom = vertexes[row];
            ListIterator<Vertex> listIteratorFrom = vertexListFrom.listIterator();
            while (listIteratorFrom.hasNext())
            {
                Vertex from = listIteratorFrom.next();
                if (from.getNature() == Nature.ns)
                {
                    int toIndex = row + from.realWord.length();
                    ListIterator<Vertex> listIteratorTo = vertexes[toIndex].listIterator();
                    while (listIteratorTo.hasNext())
                    {
                        Vertex to = listIteratorTo.next();
                        if (to.getNature() == Nature.ns)
                        {
                            // 我们不能直接改，因为很多条线路在公用指针
//                            from.realWord += to.realWord;
                            logger.info("合并【" + from.realWord + "】和【" + to.realWord + "】");
                            listIteratorFrom.set(Vertex.newAddressInstance(from.realWord + to.realWord));
//                            listIteratorTo.remove();
                            break;
                        }
                    }
                }
            }
        }
    }

}
