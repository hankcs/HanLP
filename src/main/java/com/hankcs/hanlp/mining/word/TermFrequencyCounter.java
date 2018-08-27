/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-31 9:16 PM</create-date>
 *
 * <copyright file="TermFrequencyCounter.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.word;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algorithm.MaxHeap;
import com.hankcs.hanlp.corpus.occurrence.TermFrequency;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.summary.KeywordExtractor;

import java.util.*;

/**
 * 词频统计工具
 *
 * @author hankcs
 */
public class TermFrequencyCounter extends KeywordExtractor implements Collection<TermFrequency>
{
    boolean filterStopWord;
    Map<String, TermFrequency> termFrequencyMap;

    /**
     * 构造
     *
     * @param filterStopWord 是否过滤停用词
     * @param segment        分词器
     */
    public TermFrequencyCounter(Segment segment, boolean filterStopWord)
    {
        this.filterStopWord = filterStopWord;
        this.defaultSegment = segment;
        termFrequencyMap = new TreeMap<String, TermFrequency>();
    }

    public TermFrequencyCounter()
    {
        this(HanLP.newSegment(), true);
    }

    public void add(String document)
    {
        if (document == null || document.isEmpty()) return;
        List<Term> termList = defaultSegment.seg(document);
        add(termList);
    }

    public void add(List<Term> termList)
    {
        if (filterStopWord)
        {
            filter(termList);
        }
        for (Term term : termList)
        {
            String word = term.word;
            TermFrequency frequency = termFrequencyMap.get(word);
            if (frequency == null)
            {
                frequency = new TermFrequency(word);
                termFrequencyMap.put(word, frequency);
            }
            else
            {
                frequency.increase();
            }
        }
    }

    /**
     * 取前N个高频词
     *
     * @param N
     * @return
     */
    public Collection<TermFrequency> top(int N)
    {
        MaxHeap<TermFrequency> heap = new MaxHeap<TermFrequency>(N, new Comparator<TermFrequency>()
        {
            @Override
            public int compare(TermFrequency o1, TermFrequency o2)
            {
                return o1.compareTo(o2);
            }
        });
        heap.addAll(termFrequencyMap.values());
        return heap.toList();
    }

    /**
     * 所有词汇的频次
     *
     * @return
     */
    public Collection<TermFrequency> all()
    {
        return termFrequencyMap.values();
    }

    @Override
    public int size()
    {
        return termFrequencyMap.size();
    }

    @Override
    public boolean isEmpty()
    {
        return termFrequencyMap.isEmpty();
    }

    @Override
    public boolean contains(Object o)
    {
        if (o instanceof String)
            return termFrequencyMap.containsKey(o);
        else if (o instanceof TermFrequency)
            return termFrequencyMap.containsValue(o);
        return false;
    }

    @Override
    public Iterator<TermFrequency> iterator()
    {
        return termFrequencyMap.values().iterator();
    }

    @Override
    public Object[] toArray()
    {
        return termFrequencyMap.values().toArray();
    }

    @Override
    public <T> T[] toArray(T[] a)
    {
        return termFrequencyMap.values().toArray(a);
    }

    @Override
    public boolean add(TermFrequency termFrequency)
    {
        TermFrequency tf = termFrequencyMap.get(termFrequency.getTerm());
        if (tf == null)
        {
            termFrequencyMap.put(termFrequency.getKey(), termFrequency);
            return true;
        }
        tf.increase(termFrequency.getFrequency());
        return false;
    }

    @Override
    public boolean remove(Object o)
    {
        return termFrequencyMap.remove(o) != null;
    }

    @Override
    public boolean containsAll(Collection<?> c)
    {
        for (Object o : c)
        {
            if (!contains(o))
                return false;
        }
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends TermFrequency> c)
    {
        for (TermFrequency termFrequency : c)
        {
            add(termFrequency);
        }
        return !c.isEmpty();
    }

    @Override
    public boolean removeAll(Collection<?> c)
    {
        for (Object o : c)
        {
            if (!remove(o))
                return false;
        }
        return true;
    }

    @Override
    public boolean retainAll(Collection<?> c)
    {
        return termFrequencyMap.values().retainAll(c);
    }

    @Override
    public void clear()
    {
        termFrequencyMap.clear();
    }

    /**
     * 提取关键词（非线程安全）
     *
     * @param termList
     * @param size
     * @return
     */
    @Override
    public List<String> getKeywords(List<Term> termList, int size)
    {
        clear();
        add(termList);
        Collection<TermFrequency> topN = top(size);
        List<String> r = new ArrayList<String>(topN.size());
        for (TermFrequency termFrequency : topN)
        {
            r.add(termFrequency.getTerm());
        }
        return r;
    }

    /**
     * 提取关键词（线程安全）
     *
     * @param document 文档内容
     * @param size     希望提取几个关键词
     * @return 一个列表
     */
    public static List<String> getKeywordList(String document, int size)
    {
        return new TermFrequencyCounter().getKeywords(document, size);
    }

    @Override
    public String toString()
    {
        final int max = 100;
        return top(Math.min(max, size())).toString();
    }
}
