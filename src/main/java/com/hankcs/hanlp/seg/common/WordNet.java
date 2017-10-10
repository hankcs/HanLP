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
package com.hankcs.hanlp.seg.common;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.other.CharType;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.NShort.Path.AtomNode;
import com.hankcs.hanlp.utility.MathTools;
import com.hankcs.hanlp.utility.Predefine;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class WordNet
{
    /**
     * 节点，每一行都是前缀词，跟图的表示方式不同
     */
    private LinkedList<Vertex> vertexes[];

    /**
     * 共有多少个节点
     */
    int size;

    /**
     * 原始句子
     *
     * @deprecated 应当使用数组，这样比较快
     */
    public String sentence;

    /**
     * 原始句子对应的数组
     */
    public char[] charArray;

    /**
     * 为一个句子生成空白词网
     *
     * @param sentence 句子
     */
    public WordNet(String sentence)
    {
        this(sentence.toCharArray());
    }

    public WordNet(char[] charArray)
    {
        this.charArray = charArray;
        vertexes = new LinkedList[charArray.length + 2];
        for (int i = 0; i < vertexes.length; ++i)
        {
            vertexes[i] = new LinkedList<Vertex>();
        }
        vertexes[0].add(Vertex.newB());
        vertexes[vertexes.length - 1].add(Vertex.newE());
        size = 2;
    }

    public WordNet(char[] charArray, List<Vertex> vertexList)
    {
        this.charArray = charArray;
        vertexes = new LinkedList[charArray.length + 2];
        for (int i = 0; i < vertexes.length; ++i)
        {
            vertexes[i] = new LinkedList<Vertex>();
        }
        int i = 0;
        for (Vertex vertex : vertexList)
        {
            vertexes[i].add(vertex);
            ++size;
            i += vertex.realWord.length();
        }
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
     * 强行添加，替换已有的顶点
     *
     * @param line
     * @param vertex
     */
    public void push(int line, Vertex vertex)
    {
        Iterator<Vertex> iterator = vertexes[line].iterator();
        while (iterator.hasNext())
        {
            if (iterator.next().realWord.length() == vertex.realWord.length())
            {
                iterator.remove();
                --size;
                break;
            }
        }
        vertexes[line].add(vertex);
        ++size;
    }

    /**
     * 添加顶点，同时检查此顶点是否悬孤，如果悬孤则自动补全
     *
     * @param line
     * @param vertex
     * @param wordNetAll 这是一个完全的词图
     */
    public void insert(int line, Vertex vertex, WordNet wordNetAll)
    {
        for (Vertex oldVertex : vertexes[line])
        {
            // 保证唯一性
            if (oldVertex.realWord.length() == vertex.realWord.length()) return;
        }
        vertexes[line].add(vertex);
        ++size;
        // 保证连接
        for (int l = line - 1; l > 1; --l)
        {
            if (get(l, 1) == null)
            {
                Vertex first = wordNetAll.getFirst(l);
                if (first == null) break;
                vertexes[l].add(first);
                ++size;
                if (vertexes[l].size() > 1) break;
            }
            else
            {
                break;
            }
        }
        // 首先保证这个词语可直达
        int l = line + vertex.realWord.length();
        if (get(l).size() == 0)
        {
            List<Vertex> targetLine = wordNetAll.get(l);
            if (targetLine == null || targetLine.size() == 0) return;
            vertexes[l].addAll(targetLine);
            size += targetLine.size();
        }
        // 直达之后一直往后
        for (++l; l < vertexes.length; ++l)
        {
            if (get(l).size() == 0)
            {
                Vertex first = wordNetAll.getFirst(l);
                if (first == null) break;
                vertexes[l].add(first);
                ++size;
                if (vertexes[l].size() > 1) break;
            }
            else
            {
                break;
            }
        }
    }

    /**
     * 全自动添加顶点
     *
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
     *
     * @param line 行号
     * @return 一个数组
     */
    public List<Vertex> get(int line)
    {
        return vertexes[line];
    }

    /**
     * 获取某一行的逆序迭代器
     * @param line 行号
     * @return 逆序迭代器
     */
    public Iterator<Vertex> descendingIterator(int line)
    {
        return vertexes[line].descendingIterator();
    }

    /**
     * 获取某一行的第一个节点
     *
     * @param line
     * @return
     */
    public Vertex getFirst(int line)
    {
        Iterator<Vertex> iterator = vertexes[line].iterator();
        if (iterator.hasNext()) return iterator.next();

        return null;
    }

    /**
     * 获取某一行长度为length的节点
     *
     * @param line
     * @param length
     * @return
     */
    public Vertex get(int line, int length)
    {
        for (Vertex vertex : vertexes[line])
        {
            if (vertex.realWord.length() == length)
            {
                return vertex;
            }
        }

        return null;
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
            int id = -1;
            switch (atomNode.nPOS)
            {
                case CharType.CT_CHINESE:
                    break;
                case CharType.CT_NUM:
                case CharType.CT_INDEX:
                case CharType.CT_CNUM:
                    nature = Nature.m;
                    sWord = Predefine.TAG_NUMBER;
                    id = CoreDictionary.M_WORD_ID;
                    break;
                case CharType.CT_DELIMITER:
                case CharType.CT_OTHER:
                    nature = Nature.w;
                    break;
                case CharType.CT_SINGLE://12021-2129-3121
                    nature = Nature.nx;
                    sWord = Predefine.TAG_CLUSTER;
                    id = CoreDictionary.X_WORD_ID;
                    break;
                default:
                    break;
            }
            // 这些通用符的量级都在10万左右
            add(line + offset, new Vertex(sWord, atomNode.sWord, new CoreDictionary.Attribute(nature, 10000), id));
            offset += atomNode.sWord.length();
        }
    }

    public int size()
    {
        return size;
    }

    /**
     * 获取顶点数组
     *
     * @return Vertex[] 按行优先列次之的顺序构造的顶点数组
     */
    private Vertex[] getVertexesLineFirst()
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
     *
     * @return 词图
     */
    public Graph toGraph()
    {
        Graph graph = new Graph(getVertexesLineFirst());

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

    /**
     * 清空词图
     */
    public void clear()
    {
        for (List<Vertex> vertexList : vertexes)
        {
            vertexList.clear();
        }
        size = 0;
    }

    /**
     * 获取内部顶点表格，谨慎操作！
     *
     * @return
     */
    public LinkedList<Vertex>[] getVertexes()
    {
        return vertexes;
    }
}
