/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/10 PM4:23</create-date>
 *
 * <copyright file="Document.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.corpus;

import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.classification.collections.FrequencyMap;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Map;

/**
 * 单个文档，或者说一个词袋，包含所有的词以及其频次，当然，还包括其分类。
 *
 * @author hankcs
 */
public class Document extends BagOfWordsDocument
{
    /**
     * 文档所属类目
     */
    public int category;

    /**
     * 一般用在训练集构造文档时
     * @param catalog
     * @param lexicon
     * @param category
     * @param tokenArray
     */
    public Document(Catalog catalog, Lexicon lexicon, String category, String[] tokenArray)
    {
        super();
        assert catalog != null;
        assert lexicon != null;
//        this.catalog = catalog;
//        this.lexicon = lexicon;

        // 将其转为数组类型，方便处理
        this.category = catalog.addCategory(category);
        // 统计词频
        for (int i = 0; i < tokenArray.length; i++)
        {
            tfMap.add(lexicon.addWord(tokenArray[i]));
        }
    }

    /**
     * 一般用在预测时构造文档用
     * @param wordIdTrie
     * @param tokenArray
     */
    public Document(ITrie<Integer> wordIdTrie, String[] tokenArray)
    {
        super();
        for (int i = 0; i < tokenArray.length; i++)
        {
            Integer id = wordIdTrie.get(tokenArray[i].toCharArray());
            if (id == null) continue;
            tfMap.add(id);
        }
    }

    /**
     * 一般用在测试集构造文档时使用
     * @param categoryId
     * @param wordId
     * @param category
     * @param tokenArray
     */
    public Document(Map<String, Integer> categoryId, BinTrie<Integer> wordId, String category, String[] tokenArray)
    {
        this(wordId, tokenArray);
        Integer id = categoryId.get(category);
        if (id == null) id = -1;
        this.category = id;
    }

    public Document(DataInputStream in) throws IOException
    {
        category = in.readInt();
        int size = in.readInt();
        tfMap = new FrequencyMap<Integer>();
        for (int i = 0; i < size; i++)
        {
            tfMap.put(in.readInt(), new int[]{in.readInt()});
        }
    }

    //    @Override
//    public String toString()
//    {
//        final StringBuilder sb = new StringBuilder(tfMap.size() * 5);
//        sb.append('《').append(super.toString()).append('》').append('\t');
//        sb.append(catalog.getCategory(category));
//        sb.append('\n');
//        for (Map.Entry<Integer, int[]> entry : tfMap.entrySet())
//        {
//            sb.append(lexicon.getWord(entry.getKey()));
//            sb.append('\t');
//            sb.append(entry.getValue()[0]);
//            sb.append('\n');
//        }
//        return sb.toString();
//    }

}