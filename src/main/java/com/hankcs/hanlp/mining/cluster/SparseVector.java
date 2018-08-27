/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-12 6:40 PM</create-date>
 *
 * <copyright file="SparseVector.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.cluster;

import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class SparseVector extends TreeMap<Integer, Double>
{
    @Override
    public Double get(Object key)
    {
        Double v = super.get(key);
        if (v == null) return 0.;
        return v;
    }

    /**
     * Normalize a vector.
     */
    void normalize()
    {
        double nrm = norm();
        for (Map.Entry<Integer, Double> d : entrySet())
        {
            d.setValue(d.getValue() / nrm);
        }
    }

    /**
     * Calculate a squared norm.
     */
    double norm_squared()
    {
        double sum = 0;
        for (Double point : values())
        {
            sum += point * point;
        }
        return sum;
    }

    /**
     * Calculate a norm.
     */
    double norm()
    {
        return (double) Math.sqrt(norm_squared());
    }

    /**
     * Multiply each value of  avector by a constant value.
     */
    void multiply_constant(double x)
    {
        for (Map.Entry<Integer, Double> entry : entrySet())
        {
            entry.setValue(entry.getValue() * x);
        }
    }

    /**
     * Add other vector.
     */
    void add_vector(SparseVector vec)
    {

        for (Map.Entry<Integer, Double> entry : vec.entrySet())
        {
            Double v = get(entry.getKey());
            if (v == null)
                v = 0.;
            put(entry.getKey(), v + entry.getValue());
        }
    }

    /**
     * Subtract other vector.
     */
    void sub_vector(SparseVector vec)
    {

        for (Map.Entry<Integer, Double> entry : vec.entrySet())
        {
            Double v = get(entry.getKey());
            if (v == null)
                v = 0.;
            put(entry.getKey(), v - entry.getValue());
        }
    }

//    /**
//     * Calculate the squared euclid distance between vectors.
//     */
//    double euclid_distance_squared(const Vector &vec1, const Vector &vec2)
//{
//    HashMap<VecKey, bool>::type done;
//    init_hash_map(VECTOR_EMPTY_KEY, done, vec1.size());
//    VecHashMap::const_iterator it1, it2;
//    double dist = 0;
//    for (it1 = vec1.hash_map()->begin(); it1 != vec1.hash_map()->end(); ++it1)
//    {
//        double val = vec2.get(it1->first);
//        dist += (it1->second - val) * (it1->second - val);
//        done[it1->first] = true;
//    }
//    for (it2 = vec2.hash_map()->begin(); it2 != vec2.hash_map()->end(); ++it2)
//    {
//        if (done.find(it2->first) == done.end())
//        {
//            double val = vec1.get(it2->first);
//            dist += (it2->second - val) * (it2->second - val);
//        }
//    }
//    return dist;
//}
//
//    /**
//     * Calculate the euclid distance between vectors.
//     */
//    double euclid_distance(const Vector &vec1, const Vector &vec2)
//{
//    return sqrt(euclid_distance_squared(vec1, vec2));
//}

    /**
     * Calculate the inner product value between vectors.
     */
    static double inner_product(SparseVector vec1, SparseVector vec2)
    {
        Iterator<Map.Entry<Integer, Double>> it;
        SparseVector other;
        if (vec1.size() < vec2.size())
        {
            it = vec1.entrySet().iterator();
            other = vec2;
        }
        else
        {
            it = vec2.entrySet().iterator();
            other = vec1;
        }
        double prod = 0;
        while (it.hasNext())
        {
            Map.Entry<Integer, Double> entry = it.next();
            prod += entry.getValue() * other.get(entry.getKey());
        }
        return prod;
    }

    /**
     * Calculate the cosine value between vectors.
     */
    double cosine(SparseVector vec1, SparseVector vec2)
    {
        double norm1 = vec1.norm();
        double norm2 = vec2.norm();
        double result = 0.0f;
        if (norm1 == 0 && norm2 == 0)
        {
            return result;
        }
        else
        {
            double prod = inner_product(vec1, vec2);
            result = prod / (norm1 * norm2);
            return Double.isNaN(result) ? 0.0f : result;
        }
    }

//    /**
//     * Calculate the Jaccard coefficient value between vectors.
//     */
//    double jaccard(const Vector &vec1, const Vector &vec2)
//{
//    double norm1 = vec1.norm();
//    double norm2 = vec2.norm();
//    double prod = inner_product(vec1, vec2);
//    double denom = norm1 + norm2 - prod;
//    double result = 0.0;
//    if (!denom)
//    {
//        return result;
//    }
//    else
//    {
//        result = prod / denom;
//        return isnan(result) ? 0.0 : result;
//    }
//}
}
