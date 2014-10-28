/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/28 21:51</create-date>
 *
 * <copyright file="UnknowWordDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;

import java.io.*;
import java.util.*;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * @author hankcs
 */
public class UnknownWordDictionary
{
    DoubleArrayTrie<Attribute> trie;
    protected String name;

    public UnknownWordDictionary(String name)
    {
        this.name = name;
        trie = new DoubleArrayTrie<Attribute>();
    }

    public boolean load(String path)
    {
        logger.info(name + "词典开始加载:" + path);
        List<String> wordList = new ArrayList<String>();
        List<Attribute> attributeList = new ArrayList<Attribute>();
        BufferedReader br = null;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            List<Integer> frequencyList = new LinkedList<Integer>();
            List<Integer> posList = new LinkedList<Integer>();
            line = br.readLine();
            String param[] = line.split(" ");
            frequencyList.add(Integer.valueOf(param[0]));
            posList.add(Integer.valueOf(param[1]));
            String word = param[2];
            while ((line = br.readLine()) != null)
            {
                param = line.split(" ");
                if (word.equals(param[2]))
                {
                    frequencyList.add(Integer.valueOf(param[0]));
                    posList.add(Integer.valueOf(param[1]));
                }
                else 
                {
                    wordList.add(word);
                    attributeList.add(new Attribute(frequencyList, posList));
                    frequencyList.clear();
                    posList.clear();
                    frequencyList.add(Integer.valueOf(param[0]));
                    posList.add(Integer.valueOf(param[1]));
                    word = param[2];
                }
            }
            wordList.add(word);
            attributeList.add(new Attribute(frequencyList, posList));
//            logger.trace("{}词典读入词条{} 属性{}", name, wordList.size(), attributeList.size());
            br.close();
        } catch (FileNotFoundException e)
        {
            logger.severe(name + "词典" + path + "不存在！");
            e.printStackTrace();
            return false;
        } catch (IOException e)
        {
            logger.severe(name + "词典" + path + "读取错误！");
            e.printStackTrace();
            return false;
        }

        int resultCode = trie.build(wordList, attributeList);
        logger.info(name + "词典DAT构建结果:" + resultCode);
        if (resultCode < 0)
        {
            sortListForBuildTrie(wordList, attributeList, path);
            trie = new DoubleArrayTrie<Attribute>();
            load(path);
        }
        logger.info(name + "词典加载成功:" + trie.size() + "个词条");
        return true;
    }

    public int getFrequency(String key, int pos)
    {
        Attribute attribute = trie.get(key);
        if (attribute == null) return 0;
        int index = Arrays.binarySearch(attribute.pos, pos);
        if (index < 0) return 0;
        return attribute.frequency[index];
    }

    /**
     * 接受键数组与值数组，排序以供建立trie树
     * @param wordList
     * @param attributeList
     */
    private static void sortListForBuildTrie(List<String> wordList, List<Attribute> attributeList, String path)
    {
        BinTrie<Attribute> binTrie = new BinTrie<Attribute>();
        for (int i = 0; i < wordList.size(); ++i)
        {
            binTrie.put(wordList.get(i), attributeList.get(i));
        }
        Collections.sort(wordList);
        try
        {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path)));
            for (String w : wordList)
            {
                Attribute attribute = binTrie.get(w);
                int i = 0;
                for (int frequency : attribute.frequency)
                {
                    bw.write(frequency + " " + attribute.pos[i++] + " " + w);
                    bw.newLine();
                }
            }
            bw.close();
        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        } catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public boolean IsExist(String key, int pos)
    {
        return getFrequency(key, pos) > 0;
    }

    public Attribute GetWordInfo(String key)
    {
        return trie.get(key);
    }


    static class Attribute
    {
        /**
         * 频次
         */
        int frequency[];
        /**
         * 词性
         */
        int pos[];

        Attribute(List<Integer> frequencyList, List<Integer> posList)
        {
            if (frequencyList.size() != posList.size()) throw new IllegalArgumentException("词频 词性列表大小不同");
            frequency = new int[frequencyList.size()];
            pos       = new int[posList.size()];
            int i;
            i = 0;
            for (Integer f : frequencyList)
            {
                frequency[i++] = f;
            }
            i = 0;
            for (Integer p : posList)
            {
                pos[i++] = p;
            }
        }
    }
}
