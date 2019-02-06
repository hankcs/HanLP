/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.structures;


import java.util.ArrayList;

/**
 * CoNLL中的一个句子
 */
public class Sentence implements Comparable
{
    /**
     * 词语id
     */
    private int[] words;
    /**
     * 词性
     */
    private int[] tags;

    private int[] brownCluster4thPrefix;
    private int[] brownCluster6thPrefix;
    private int[] brownClusterFullString;


    public Sentence(ArrayList<Integer> tokens, ArrayList<Integer> pos, ArrayList<Integer> brownCluster4thPrefix, ArrayList<Integer> brownCluster6thPrefix, ArrayList<Integer> brownClusterFullString)
    {
        words = new int[tokens.size()];
        tags = new int[tokens.size()];
        this.brownCluster4thPrefix = new int[tokens.size()];
        this.brownCluster6thPrefix = new int[tokens.size()];
        this.brownClusterFullString = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++)
        {
            words[i] = tokens.get(i);
            tags[i] = pos.get(i);
            this.brownCluster4thPrefix[i] = brownCluster4thPrefix.get(i);
            this.brownCluster6thPrefix[i] = brownCluster6thPrefix.get(i);
            this.brownClusterFullString[i] = brownClusterFullString.get(i);
        }
    }

    public int size()
    {
        return words.length;
    }

    public int posAt(int position)
    {
        if (position == 0)
            return 0;

        return tags[position - 1];
    }

    public int[] getWords()
    {
        return words;
    }

    public int[] getTags()
    {
        return tags;
    }


    public int[] getBrownCluster4thPrefix()
    {
        return brownCluster4thPrefix;
    }


    public int[] getBrownCluster6thPrefix()
    {
        return brownCluster6thPrefix;
    }

    public int[] getBrownClusterFullString()
    {
        return brownClusterFullString;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (obj instanceof Sentence)
        {
            Sentence sentence = (Sentence) obj;
            if (sentence.words.length != words.length)
                return false;
            for (int i = 0; i < sentence.words.length; i++)
            {
                if (sentence.words[i] != words[i])
                    return false;
                if (sentence.tags[i] != tags[i])
                    return false;
            }
            return true;
        }
        return false;
    }

    @Override
    public int compareTo(Object o)
    {
        if (equals(o))
            return 0;
        return hashCode() - o.hashCode();
    }

    @Override
    public int hashCode()
    {
        int hash = 0;
        for (int tokenId = 0; tokenId < words.length; tokenId++)
        {
            hash ^= (words[tokenId] * tags[tokenId]);
        }
        return hash;
    }

}
