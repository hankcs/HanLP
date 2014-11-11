/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/10 12:42</create-date>
 *
 * <copyright file="WordDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;


import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 核心词典
 *
 * @author He Han
 */
public class CoreDictionary
{
    static DoubleArrayTrie<Attribute> trie = new DoubleArrayTrie<Attribute>();
    public final static String path = HanLP.Config.CoreDictionaryPath;
    public static final int totalFrequency = 221894;

    //    public final static String path = "data/dictionary/CoreDictionary.txt";
    // 自动加载词典
    static
    {
        long start = System.currentTimeMillis();
        if (!load(path))
        {
            System.err.printf("核心词典%s加载失败\n", path);
            System.exit(-1);
        }
        else
        {
            logger.info(path + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        }
    }


    public static boolean load(String path)
    {
        logger.info("核心词典开始加载:" + path);
        if (loadDat(path)) return true;
        List<String> wordList = new ArrayList<String>();
        List<Attribute> attributeList = new ArrayList<Attribute>();
        BufferedReader br = null;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            int MAX_FREQUENCY = 0;
            long start = System.currentTimeMillis();
            while ((line = br.readLine()) != null)
            {
                String param[] = line.split(" ");
                wordList.add(param[0]);
                int natureCount = (param.length - 1) / 2;
                Attribute attribute = new Attribute(natureCount);
                for (int i = 0; i < natureCount; ++i)
                {
                    attribute.nature[i] = Enum.valueOf(Nature.class, param[1 + 2 * i]);
                    attribute.frequency[i] = Integer.parseInt(param[2 + 2 * i]);
                    attribute.totalFrequency += attribute.frequency[i];
                }
                attributeList.add(attribute);
                MAX_FREQUENCY += attribute.totalFrequency;
            }
            logger.info("核心词典读入词条" + wordList.size() + " 全部频次" + MAX_FREQUENCY + "，耗时" + (System.currentTimeMillis() - start) + "ms");
            br.close();
            logger.info("核心词典DAT构建结果:" + trie.build(wordList, attributeList));
            logger.info("核心词典加载成功:" + trie.size() + "个词条");
            // 缓存成dat文件，下次加载会快很多
            if (!trie.save(path + ".trie.dat")) return false;
            // 缓存值文件
            try
            {
                DataOutputStream out = new DataOutputStream(new FileOutputStream(path + ".value.dat"));
                out.writeInt(attributeList.size());
                for (Attribute attribute : attributeList)
                {
                    out.writeInt(attribute.totalFrequency);
                    out.writeInt(attribute.nature.length);
                    for (int i = 0; i < attribute.nature.length; ++i)
                    {
                        out.writeInt(attribute.nature[i].ordinal());
                        out.writeInt(attribute.frequency[i]);
                    }
                }
                out.close();
            }
            catch (Exception e)
            {
                logger.warning("保存失败" + e);
                return false;
            }
        }
        catch (FileNotFoundException e)
        {
            logger.warning("核心词典" + path + "不存在！" + e);
            return false;
        }
        catch (IOException e)
        {
            logger.warning("核心词典" + path + "读取错误！" + e);
            return false;
        }

        return true;
    }

    /**
     * 从磁盘加载双数组
     *
     * @param path
     * @return
     */
    static boolean loadDat(String path)
    {
        try
        {
            byte[] bytes = IOUtil.readBytes(path + ".value.dat");
            if (bytes == null) return false;
            int index = 0;
            int size = TextUtility.bytesHighFirstToInt(bytes, index);
            index += 4;
            Attribute[] attributes = new Attribute[size];
            final Nature[] natureIndexArray = Nature.values();
            for (int i = 0; i < size; ++i)
            {
                // 第一个是全部频次，第二个是词性个数
                int currentTotalFrequency = TextUtility.bytesHighFirstToInt(bytes, index);
                index += 4;
                int length = TextUtility.bytesHighFirstToInt(bytes, index);
                index += 4;
                attributes[i] = new Attribute(length);
                attributes[i].totalFrequency = currentTotalFrequency;
                for (int j = 0; j < length; ++j)
                {
                    attributes[i].nature[j] = natureIndexArray[TextUtility.bytesHighFirstToInt(bytes, index)];
                    index += 4;
                    attributes[i].frequency[j] = TextUtility.bytesHighFirstToInt(bytes, index);
                    index += 4;
                }
            }
            logger.info("值" + path + ".value.dat" + "加载完毕");
            if (!trie.load(path + ".trie.dat", attributes)) return false;
            logger.info("双数组" + path + ".trie.dat" + "加载完毕");
        }
        catch (Exception e)
        {
            logger.warning("读取失败，问题发生在" + e);
            return false;
        }
        return true;
    }

    public static Attribute GetWordInfo(String key)
    {
        return trie.get(key);
    }

    /**
     * 获取词频
     *
     * @param term
     * @return
     */
    public static int getTermFrequency(String term)
    {
        Attribute attribute = GetWordInfo(term);
        if (attribute == null) return 0;
        return attribute.totalFrequency;
    }

    /**
     * 词典是否包含这个词语
     *
     * @param key
     * @return
     */
    public static boolean contains(String key)
    {
        return trie.get(key) != null;
    }

    public static BaseSearcher getSearcher(String text)
    {
        return new Searcher(text);
    }

    public static class Searcher extends BaseSearcher<Attribute>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        private LinkedList<Map.Entry<String, Attribute>> entryList;

        protected Searcher(char[] c)
        {
            super(c);
        }

        protected Searcher(String text)
        {
            super(text);
            entryList = new LinkedList<Map.Entry<String, Attribute>>();
        }

        @Override
        public Map.Entry<String, Attribute> next()
        {
            // 保证首次调用找到一个词语
            while (entryList.size() == 0 && begin < c.length)
            {
                entryList = trie.commonPrefixSearchWithValue(c, begin);
                ++begin;
            }
            // 之后调用仅在缓存用完的时候调用一次
            if (entryList.size() == 0 && begin < c.length)
            {
                entryList = trie.commonPrefixSearchWithValue(c, begin);
                ++begin;
            }
            if (entryList.size() == 0)
            {
                return null;
            }
            Map.Entry<String, Attribute> result = entryList.getLast();
            entryList.removeLast();
            offset = begin - 1;
            return result;
        }
    }

    /**
     * 核心词典中的词属性
     */
    static public class Attribute implements Serializable
    {
        /**
         * 词性列表
         */
        public Nature nature[];
        /**
         * 词性对应的词频
         */
        public int frequency[];

        public int totalFrequency;

        // 几个预定义的变量

//        public static Attribute NUMBER = new Attribute()

        public Attribute(int size)
        {
            nature = new Nature[size];
            frequency = new int[size];
        }

        public Attribute(Nature[] nature, int[] frequency)
        {
            this.nature = nature;
            this.frequency = frequency;
        }

        public Attribute(Nature nature, int frequency)
        {
            this(1);
            this.nature[0] = nature;
            this.frequency[0] = frequency;
            totalFrequency = frequency;
        }

        public Attribute(Nature[] nature, int[] frequency, int totalFrequency)
        {
            this.nature = nature;
            this.frequency = frequency;
            this.totalFrequency = totalFrequency;
        }

        /**
         * 使用单个词性，默认词频1000构造
         *
         * @param nature
         */
        public Attribute(Nature nature)
        {
            this(nature, 1000);
        }

        /**
         * 获取词性的词频
         *
         * @param nature 字符串词性
         * @return 词频
         * @deprecated 推荐使用Nature参数！
         */
        public int getNatureFrequency(String nature)
        {
            try
            {
                Nature pos = Enum.valueOf(Nature.class, nature);
                return getNatureFrequency(pos);
            }
            catch (IllegalArgumentException e)
            {
                return 0;
            }
        }

        /**
         * 获取词性的词频
         *
         * @param nature 词性
         * @return 词频
         */
        public int getNatureFrequency(final Nature nature)
        {
            int result = 0;
            int i = 0;
            for (Nature pos : this.nature)
            {
                if (nature == pos)
                {
                    return frequency[i];
                }
                ++i;
            }
            return result;
        }

        @Override
        public String toString()
        {
            final StringBuilder sb = new StringBuilder();
            for (int i = 0; i < nature.length; ++i)
            {
                sb.append(nature[i]).append(' ').append(frequency[i]);
            }
            return sb.toString();
        }
    }
}
