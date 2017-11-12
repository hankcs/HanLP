/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-06-20 PM1:38</create-date>
 *
 * <copyright file="DocVectorModel.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.word2vec;


import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NotionalTokenizer;

import java.util.*;

/**
 * 文档向量模型
 *
 * @author hankcs
 */
public class DocVectorModel extends AbstractVectorModel<Integer>
{
    private final WordVectorModel wordVectorModel;

    public DocVectorModel(WordVectorModel wordVectorModel)
    {
        super();
        this.wordVectorModel = wordVectorModel;
    }

    /**
     * 添加文档
     *
     * @param id      文档id
     * @param content 文档内容
     * @return 文档向量
     */
    public Vector addDocument(int id, String content)
    {
        Vector result = query(content);
        storage.put(id, result);
        return result;
    }


    /**
     * 查询最相似的前10个文档
     *
     * @param query 查询语句（或者说一个文档的内容）
     * @return
     */
    public List<Map.Entry<Integer, Float>> nearest(String query)
    {
        return queryNearest(query, 10);
    }


    /**
     * 将一个文档转为向量
     *
     * @param content 文档
     * @return 向量
     */
    public Vector query(String content)
    {
        if (content == null || content.length() == 0) return null;
        List<Term> termList = NotionalTokenizer.segment(content);
        Vector result = new Vector(dimension());
        int n = 0;
        for (Term term : termList)
        {
            Vector vector = wordVectorModel.vector(term.word);
            if (vector == null)
            {
                continue;
            }
            ++n;
            result.addToSelf(vector);
        }
        if (n == 0)
        {
            return null;
        }
        result.normalize();
        return result;
    }

    @Override
    public int dimension()
    {
        return wordVectorModel.dimension();
    }

    /**
     * 文档相似度计算
     * @param what
     * @param with
     * @return
     */
    public float similarity(String what, String with)
    {
        Vector A = query(what);
        if (A == null) return -1f;
        Vector B = query(with);
        if (B == null) return -1f;
        return A.cosineForUnitVector(B);
    }
}
