/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-18 11:11 PM</create-date>
 *
 * <copyright file="DemoTextClustering.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.mining.cluster.ClusterAnalyzer;

import static com.hankcs.demo.DemoTextClassification.CORPUS_FOLDER;

/**
 * @author hankcs
 */
public class DemoTextClusteringFMeasure
{
    public static void main(String[] args)
    {
        for (String algorithm : new String[]{"kmeans", "repeated bisection"})
        {
            System.out.printf("%s F1=%.2f\n", algorithm, ClusterAnalyzer.evaluate(CORPUS_FOLDER, algorithm) * 100);
        }
    }
}
