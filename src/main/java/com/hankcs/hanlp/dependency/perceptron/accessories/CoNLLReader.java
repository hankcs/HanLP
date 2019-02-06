/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.accessories;

import com.hankcs.hanlp.dependency.perceptron.structures.IndexMaps;
import com.hankcs.hanlp.dependency.perceptron.structures.Sentence;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.CompactTree;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Instance;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class CoNLLReader
{
    /**
     * An object for reading the CoNLL file
     */
    BufferedReader fileReader;

    /**
     * Initializes the file reader
     *
     * @param filePath Path to the file
     * @throws Exception If the file path is not correct or there are not enough permission to read the file
     */
    public CoNLLReader(String filePath) throws FileNotFoundException
    {
        fileReader = new BufferedReader(new FileReader(filePath));
    }

    /**
     * 读取CoNLL文件，创建索引
     *
     * @param conllPath
     * @param labeled
     * @param lowercased
     * @param clusterFile
     * @return
     * @throws Exception
     */
    public static IndexMaps createIndices(String conllPath, boolean labeled, boolean lowercased, String clusterFile) throws IOException
    {
        HashMap<String, Integer> wordMap = new HashMap<String, Integer>();
        HashMap<Integer, Integer> labels = new HashMap<Integer, Integer>();
        HashMap<String, Integer> clusterMap = new HashMap<String, Integer>();
        HashMap<Integer, Integer> cluster4Map = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> cluster6Map = new HashMap<Integer, Integer>();

        String rootString = "ROOT";

        wordMap.put("ROOT", 0);
        labels.put(0, 0);

        // 所有label的id必须从零开始并且连续
        BufferedReader reader = new BufferedReader(new FileReader(conllPath));
        String line;
        while ((line = reader.readLine()) != null)
        {
            String[] args = line.trim().split("\t");
            if (args.length > 7)
            {
                String label = args[7];
                int head = Integer.parseInt(args[6]);
                if (head == 0)
                    rootString = label;

                if (!labeled)
                    label = "~";
                else if (label.equals("_"))
                    label = "-";

                if (!wordMap.containsKey(label))
                {
                    labels.put(wordMap.size(), labels.size());
                    wordMap.put(label, wordMap.size());
                }
            }
        }

        reader = new BufferedReader(new FileReader(conllPath));
        while ((line = reader.readLine()) != null)
        {
            String[] cells = line.trim().split("\t");
            if (cells.length > 7)
            {
                String pos = cells[3];
                if (!wordMap.containsKey(pos))
                {
                    wordMap.put(pos, wordMap.size());
                }
            }
        }

        if (clusterFile.length() > 0)
        {
            reader = new BufferedReader(new FileReader(clusterFile));
            while ((line = reader.readLine()) != null)
            {
                String[] cells = line.trim().split("\t");
                if (cells.length > 2)
                {
                    String cluster = cells[0];
                    String word = cells[1];
                    String prefix4 = cluster.substring(0, Math.min(4, cluster.length()));
                    String prefix6 = cluster.substring(0, Math.min(6, cluster.length()));
                    int clusterId = wordMap.size();

                    if (!wordMap.containsKey(cluster))
                    {
                        clusterMap.put(word, wordMap.size());
                        wordMap.put(cluster, wordMap.size());
                    }
                    else
                    {
                        clusterId = wordMap.get(cluster);
                        clusterMap.put(word, clusterId);
                    }

                    int pref4Id = wordMap.size();
                    if (!wordMap.containsKey(prefix4))
                    {
                        wordMap.put(prefix4, wordMap.size());
                    }
                    else
                    {
                        pref4Id = wordMap.get(prefix4);
                    }

                    int pref6Id = wordMap.size();
                    if (!wordMap.containsKey(prefix6))
                    {
                        wordMap.put(prefix6, wordMap.size());
                    }
                    else
                    {
                        pref6Id = wordMap.get(prefix6);
                    }

                    cluster4Map.put(clusterId, pref4Id);
                    cluster6Map.put(clusterId, pref6Id);
                }
            }
        }

        reader = new BufferedReader(new FileReader(conllPath));
        while ((line = reader.readLine()) != null)
        {
            String[] cells = line.trim().split("\t");
            if (cells.length > 7)
            {
                String word = cells[1];
                if (lowercased)
                    word = word.toLowerCase();
                if (!wordMap.containsKey(word))
                {
                    wordMap.put(word, wordMap.size());
                }
            }
        }

        return new IndexMaps(wordMap, labels, rootString, cluster4Map, cluster6Map, clusterMap);
    }

    /**
     * 读取句子
     *
     * @param limit             最大多少句
     * @param keepNonProjective 保留非投影
     * @param labeled
     * @param rootFirst         是否把root放到最前面
     * @param lowerCased
     * @param maps              feature id map
     * @return
     * @throws Exception
     */
    public ArrayList<Instance> readData(int limit, boolean keepNonProjective, boolean labeled, boolean rootFirst, boolean lowerCased, IndexMaps maps) throws IOException
    {
        HashMap<String, Integer> wordMap = maps.getWordId();
        ArrayList<Instance> instanceList = new ArrayList<Instance>();

        String line;
        ArrayList<Integer> tokens = new ArrayList<Integer>();
        ArrayList<Integer> tags = new ArrayList<Integer>();
        ArrayList<Integer> cluster4Ids = new ArrayList<Integer>();
        ArrayList<Integer> cluster6Ids = new ArrayList<Integer>();
        ArrayList<Integer> clusterIds = new ArrayList<Integer>();

        HashMap<Integer, Edge> goldDependencies = new HashMap<Integer, Edge>();
        int sentenceCounter = 0;
        while ((line = fileReader.readLine()) != null)
        {
            line = line.trim();
            if (line.length() == 0) // 句子分隔空白行
            {
                if (tokens.size() > 0)
                {
                    sentenceCounter++;
                    if (!rootFirst)
                    {
                        for (Edge edge : goldDependencies.values())
                        {
                            if (edge.headIndex == 0)
                                edge.headIndex = tokens.size() + 1;
                        }
                        tokens.add(0);
                        tags.add(0);
                        cluster4Ids.add(0);
                        cluster6Ids.add(0);
                        clusterIds.add(0);
                    }
                    Sentence currentSentence = new Sentence(tokens, tags, cluster4Ids, cluster6Ids, clusterIds);
                    Instance instance = new Instance(currentSentence, goldDependencies);
                    if (keepNonProjective || !instance.isNonprojective())
                        instanceList.add(instance);
                    goldDependencies = new HashMap<Integer, Edge>();
                    tokens = new ArrayList<Integer>();
                    tags = new ArrayList<Integer>();
                    cluster4Ids = new ArrayList<Integer>();
                    cluster6Ids = new ArrayList<Integer>();
                    clusterIds = new ArrayList<Integer>();
                }
                else
                {
                    goldDependencies = new HashMap<Integer, Edge>();
                    tokens = new ArrayList<Integer>();
                    tags = new ArrayList<Integer>();
                    cluster4Ids = new ArrayList<Integer>();
                    cluster6Ids = new ArrayList<Integer>();
                    clusterIds = new ArrayList<Integer>();
                }
                if (sentenceCounter >= limit)
                {
                    System.out.println("buffer full..." + instanceList.size());
                    break;
                }
            }
            else
            {
                String[] cells = line.split("\t");
                if (cells.length < 8)
                    throw new IllegalArgumentException("invalid conll format");
                int wordIndex = Integer.parseInt(cells[0]);
                String word = cells[1].trim();
                if (lowerCased)
                    word = word.toLowerCase();
                String pos = cells[3].trim();

                int wi = getId(word, wordMap);
                int pi = getId(pos, wordMap);

                tags.add(pi);
                tokens.add(wi);

                int headIndex = Integer.parseInt(cells[6]);
                String relation = cells[7];
                if (!labeled)
                    relation = "~";
                else if (relation.equals("_"))
                    relation = "-";

                if (headIndex == 0)
                    relation = "ROOT";

                int ri = getId(relation, wordMap);
                if (headIndex == -1)
                    ri = -1;

                int[] ids = maps.clusterId(word);
                clusterIds.add(ids[0]);
                cluster4Ids.add(ids[1]);
                cluster6Ids.add(ids[2]);

                if (headIndex >= 0)
                    goldDependencies.put(wordIndex, new Edge(headIndex, ri));
            }
        }
        if (tokens.size() > 0)
        {
            if (!rootFirst)
            {
                for (int gold : goldDependencies.keySet())
                {
                    if (goldDependencies.get(gold).headIndex == 0)
                        goldDependencies.get(gold).headIndex = goldDependencies.size() + 1;
                }
                tokens.add(0);
                tags.add(0);
                cluster4Ids.add(0);
                cluster6Ids.add(0);
                clusterIds.add(0);
            }
            sentenceCounter++;
            Sentence currentSentence = new Sentence(tokens, tags, cluster4Ids, cluster6Ids, clusterIds);
            instanceList.add(new Instance(currentSentence, goldDependencies));
        }

        return instanceList;
    }

    private static int getId(String word, HashMap<String, Integer> wordMap)
    {
        return getId(word, wordMap, -1);
    }

    private static int getId(String word, HashMap<String, Integer> wordMap, int defaultValue)
    {
        Integer id = wordMap.get(word);
        if (id == null) return defaultValue;
        return id;
    }

    public ArrayList<CompactTree> readStringData() throws IOException
    {
        ArrayList<CompactTree> treeSet = new ArrayList<CompactTree>();

        String line;
        ArrayList<String> tags = new ArrayList<String>();

        HashMap<Integer, Pair<Integer, String>> goldDependencies = new HashMap<Integer, Pair<Integer, String>>();
        while ((line = fileReader.readLine()) != null)
        {
            line = line.trim();
            if (line.length() == 0)
            {
                if (tags.size() >= 1)
                {
                    CompactTree goldConfiguration = new CompactTree(goldDependencies, tags);
                    treeSet.add(goldConfiguration);
                }
                tags = new ArrayList<String>();
                goldDependencies = new HashMap<Integer, Pair<Integer, String>>();
            }
            else
            {
                String[] splitLine = line.split("\t");
                if (splitLine.length < 8)
                    throw new IllegalArgumentException("wrong file format");
                int wordIndex = Integer.parseInt(splitLine[0]);
                String pos = splitLine[3].trim();

                tags.add(pos);

                int headIndex = Integer.parseInt(splitLine[6]);
                String relation = splitLine[7];

                if (headIndex == 0)
                {
                    relation = "ROOT";
                }

                if (pos.length() > 0)
                    goldDependencies.put(wordIndex, new Pair<Integer, String>(headIndex, relation));
            }
        }


        if (tags.size() > 0)
        {
            treeSet.add(new CompactTree(goldDependencies, tags));
        }

        return treeSet;
    }

}
