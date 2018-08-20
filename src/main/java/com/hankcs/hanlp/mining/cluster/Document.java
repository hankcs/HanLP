/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-12 7:15 PM</create-date>
 *
 * <copyright file="Document.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.cluster;

import java.util.HashMap;
import java.util.Map;

/**
 * @author hankcs
 */
public class Document<K>
{
    K id_;    /// the identifier of a document
    SparseVector feature_;  /// feature vector of a document

    public Document(K id_, SparseVector feature_)
    {
        this.id_ = id_;
        this.feature_ = feature_;
    }

    public Document(K id_)
    {
        this(id_, new SparseVector());
    }

    /**
     * Get an identifier.
     *
     * @return an identifier
     */
    K id()
    {
        return id_;
    }

    /**
     * Get the pointer of a feature vector
     *
     * @return the pointer of a feature vector
     */
    SparseVector feature()
    {
        return feature_;
    }


    /**
     * Add a feature.
     *
     * @param key   the key of a feature
     * @param value the value of a feature
     */
    void add_feature(int key, double value)
    {
        feature_.put(key, value);
    }

    /**
     * Set features.
     *
     * @param feature a feature vector
     */
    void set_features(SparseVector feature)
    {
        feature_ = feature;
    }

    /**
     * Clear features.
     */
    void clear()
    {
        feature_.clear();
    }

    /**
     * Apply IDF(inverse document frequency) weighting.
     *
     * @param df    document frequencies
     * @param ndocs the number of documents
     */
    void idf(HashMap<Integer, Integer> df, int ndocs)
    {
        for (Map.Entry<Integer, Double> entry : feature_.entrySet())
        {
            Integer denom = df.get(entry.getKey());
            if (denom == null) denom = 1;
            entry.setValue((double) (entry.getValue() * Math.log(ndocs / denom)));
        }
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Document<?> document = (Document<?>) o;

        return id_ != null ? id_.equals(document.id_) : document.id_ == null;
    }

    @Override
    public int hashCode()
    {
        return id_ != null ? id_.hashCode() : 0;
    }
}
