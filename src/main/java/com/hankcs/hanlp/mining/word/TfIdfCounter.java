/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-12 PM4:22</create-date>
 *
 * <copyright file="TfIdfKeyword.java" company="码农场">
 * Copyright (c) 2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.word;

import com.hankcs.hanlp.algorithm.MaxHeap;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.summary.KeywordExtractor;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * TF-IDF统计工具兼关键词提取工具
 *
 * @author hankcs
 */
public class TfIdfCounter extends KeywordExtractor
{
    private boolean filterStopWord;
    private Map<Object, Map<String, Double>> tfMap;
    private Map<Object, Map<String, Double>> tfidfMap;
    private Map<String, Double> idf;

    public TfIdfCounter()
    {
        this(true);
    }

    public TfIdfCounter(boolean filterStopWord)
    {
        this(StandardTokenizer.SEGMENT, filterStopWord);
    }

    public TfIdfCounter(Segment defaultSegment, boolean filterStopWord)
    {
        super(defaultSegment);
        this.filterStopWord = filterStopWord;
        tfMap = new HashMap<Object, Map<String, Double>>();
    }

    public TfIdfCounter(Segment defaultSegment)
    {
        this(defaultSegment, true);
    }

    @Override
    public List<String> getKeywords(List<Term> termList, int size)
    {
        List<Map.Entry<String, Double>> entryList = getKeywordsWithTfIdf(termList, size);
        List<String> r = new ArrayList<String>(entryList.size());
        for (Map.Entry<String, Double> entry : entryList)
        {
            r.add(entry.getKey());
        }

        return r;
    }

    public List<Map.Entry<String, Double>> getKeywordsWithTfIdf(String document, int size)
    {
        return getKeywordsWithTfIdf(preprocess(document), size);
    }


    public List<Map.Entry<String, Double>> getKeywordsWithTfIdf(List<Term> termList, int size)
    {
        if (idf == null)
            compute();

        Map<String, Double> tfIdf = TfIdf.tfIdf(TfIdf.tf(convert(termList)), idf);
        return topN(tfIdf, size);
    }

    public void add(Object id, List<Term> termList)
    {
        List<String> words = convert(termList);
        Map<String, Double> tf = TfIdf.tf(words);
        tfMap.put(id, tf);
        idf = null;
    }

    private static List<String> convert(List<Term> termList)
    {
        List<String> words = new ArrayList<String>(termList.size());
        for (Term term : termList)
        {
            words.add(term.word);
        }
        return words;
    }

    public void add(List<Term> termList)
    {
        add(tfMap.size(), termList);
    }

    /**
     * 添加文档
     *
     * @param id   文档id
     * @param text 文档内容
     */
    public void add(Object id, String text)
    {
        List<Term> termList = preprocess(text);
        add(id, termList);
    }

    private List<Term> preprocess(String text)
    {
        List<Term> termList = defaultSegment.seg(text);
        if (filterStopWord)
        {
            filter(termList);
        }
        return termList;
    }

    /**
     * 添加文档，自动分配id
     *
     * @param text
     */
    public int add(String text)
    {
        int id = tfMap.size();
        add(id, text);
        return id;
    }
    
    /**
     * 加载自定义idf文件
     *
     * @param idfPath
     */
    public void loadIdfFile(String idfPath){
        String line = null;
        boolean first = true;
        try
        {
            idf  = new HashMap<String, Double>();
            BufferedReader bw = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(idfPath), "UTF-8"));
            while ((line = bw.readLine()) != null)
            {
                if (first)
                {
                    first = false;
                    if (!line.isEmpty() && line.charAt(0) == '\uFEFF')
                        line = line.substring(1);
                }
                String lineValue[] = line.split(" ");
                idf.put(lineValue[0],Double.valueOf( lineValue[1]));
            }
            bw.close();
        }
        catch (Exception e)
        {
            logger.warning("加载" + idfPath + "失败，" + e);
            throw new RuntimeException("载入反文档词频文件" + idfPath + "失败");
        }

    }

    public Map<Object, Map<String, Double>> compute()
    {
        // 如果没有加载idf文件，则通过tf计算
        if(idf==null) {
            idf = TfIdf.idfFromTfs(tfMap.values());
        }
        tfidfMap = new HashMap<Object, Map<String, Double>>(idf.size());
        for (Map.Entry<Object, Map<String, Double>> entry : tfMap.entrySet())
        {
            Map<String, Double> tfidf = TfIdf.tfIdf(entry.getValue(), idf);
            tfidfMap.put(entry.getKey(), tfidf);
        }
        return tfidfMap;
    }

    public List<Map.Entry<String, Double>> getKeywordsOf(Object id)
    {
        return getKeywordsOf(id, 10);
    }


    public List<Map.Entry<String, Double>> getKeywordsOf(Object id, int size)
    {
        Map<String, Double> tfidfs = tfidfMap.get(id);
        if (tfidfs == null) return null;

        return topN(tfidfs, size);
    }

    private List<Map.Entry<String, Double>> topN(Map<String, Double> tfidfs, int size)
    {
        MaxHeap<Map.Entry<String, Double>> heap = new MaxHeap<Map.Entry<String, Double>>(size, new Comparator<Map.Entry<String, Double>>()
        {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2)
            {
                return o1.getValue().compareTo(o2.getValue());
            }
        });

        heap.addAll(tfidfs.entrySet());
        return heap.toList();
    }

    public Set<Object> documents()
    {
        return tfMap.keySet();
    }

    public Map<Object, Map<String, Double>> getTfMap()
    {
        return tfMap;
    }

    public List<Map.Entry<String, Double>> sortedAllTf()
    {
        return sort(allTf());
    }

    public List<Map.Entry<String, Integer>> sortedAllTfInt()
    {
        return doubleToInteger(sortedAllTf());
    }

    public Map<String, Double> allTf()
    {
        Map<String, Double> result = new HashMap<String, Double>();
        for (Map<String, Double> d : tfMap.values())
        {
            for (Map.Entry<String, Double> tf : d.entrySet())
            {
                Double f = result.get(tf.getKey());
                if (f == null)
                {
                    result.put(tf.getKey(), tf.getValue());
                }
                else
                {
                    result.put(tf.getKey(), f + tf.getValue());
                }
            }
        }

        return result;
    }

    private static List<Map.Entry<String, Double>> sort(Map<String, Double> map)
    {
        List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String, Double>>()
        {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2)
            {
                return o2.getValue().compareTo(o1.getValue());
            }
        });

        return list;
    }

    private static List<Map.Entry<String, Integer>> doubleToInteger(List<Map.Entry<String, Double>> list)
    {
        List<Map.Entry<String, Integer>> result = new ArrayList<Map.Entry<String, Integer>>(list.size());
        for (Map.Entry<String, Double> entry : list)
        {
            result.add(new AbstractMap.SimpleEntry<String, Integer>(entry.getKey(), entry.getValue().intValue()));
        }

        return result;
    }
}
