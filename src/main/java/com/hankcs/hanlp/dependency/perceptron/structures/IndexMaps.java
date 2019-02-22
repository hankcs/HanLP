/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.structures;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * 将所有字符串混到一起赋予id的结构
 */
public class IndexMaps implements Serializable
{
    /**
     * ROOT
     */
    public final String rootString;
    /**
     * uid to word
     */
    public String[] idWord;
    /**
     * word(包含pos、label、cluster) to uid
     */
    private HashMap<String, Integer> wordId;
    /**
     * label id to uid，所有label的id必须从零开始并且连续
     */
    private HashMap<Integer, Integer> labels;
    /**
     * cluster id to prefix 4 id
     */
    private HashMap<Integer, Integer> brown4Clusters;
    private HashMap<Integer, Integer> brown6Clusters;
    /**
     * word to cluster id
     */
    private HashMap<String, Integer> brownFullClusters;

    public IndexMaps(HashMap<String, Integer> wordId, HashMap<Integer, Integer> labels, String rootString,
                     HashMap<Integer, Integer> brown4Clusters, HashMap<Integer, Integer> brown6Clusters, HashMap<String, Integer> brownFullClusters)
    {
        this.wordId = wordId;
        this.labels = labels;

        idWord = new String[wordId.size() + 1];
        idWord[0] = "ROOT";

        for (String word : wordId.keySet())
        {
            idWord[wordId.get(word)] = word;
        }

        this.brown4Clusters = brown4Clusters;
        this.brown6Clusters = brown6Clusters;
        this.brownFullClusters = brownFullClusters;
        this.rootString = rootString;
    }

    /**
     * 将句子中的字符串转换为id
     *
     * @param words
     * @param posTags
     * @param rootFirst
     * @param lowerCased
     * @return
     */
    public Sentence makeSentence(String[] words, String[] posTags, boolean rootFirst, boolean lowerCased)
    {
        ArrayList<Integer> tokens = new ArrayList<Integer>();
        ArrayList<Integer> tags = new ArrayList<Integer>();
        ArrayList<Integer> bc4 = new ArrayList<Integer>();
        ArrayList<Integer> bc6 = new ArrayList<Integer>();
        ArrayList<Integer> bcf = new ArrayList<Integer>();

        int i = 0;
        for (String word : words)
        {
            if (word.length() == 0)
                continue;
            String lowerCaseWord = word.toLowerCase();
            if (lowerCased)
                word = lowerCaseWord;

            int[] clusterIDs = clusterId(word);
            bcf.add(clusterIDs[0]);
            bc4.add(clusterIDs[1]);
            bc6.add(clusterIDs[2]);

            String pos = posTags[i];

            int wi = -1;
            if (wordId.containsKey(word))
                wi = wordId.get(word);

            int pi = -1;
            if (wordId.containsKey(pos))
                pi = wordId.get(pos);

            tokens.add(wi);
            tags.add(pi);

            i++;
        }

        if (!rootFirst)
        {
            tokens.add(0);
            tags.add(0);
            bcf.add(0);
            bc6.add(0);
            bc4.add(0);
        }

        return new Sentence(tokens, tags, bc4, bc6, bcf);
    }

    public HashMap<String, Integer> getWordId()
    {
        return wordId;
    }

    /**
     * 依存关系
     *
     * @return
     */
    public HashMap<Integer, Integer> getLabels()
    {
        return labels;
    }

    /**
     * 获取聚类id
     *
     * @param word
     * @return
     */
    public int[] clusterId(String word)
    {
        int[] ids = new int[3];
        ids[0] = -100;
        ids[1] = -100;
        ids[2] = -100;
        if (brownFullClusters.containsKey(word))
            ids[0] = brownFullClusters.get(word);

        if (ids[0] > 0)
        {
            ids[1] = brown4Clusters.get(ids[0]);
            ids[2] = brown6Clusters.get(ids[0]);
        }
        return ids;
    }

    public boolean hasClusters()
    {
        if (brownFullClusters != null && brownFullClusters.size() > 0)
            return true;
        return false;
    }
}
