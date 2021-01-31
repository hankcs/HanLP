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
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.utility.LexiconUtility;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 用户自定义词典<br>
 * 注意自定义词典的动态增删改不是线程安全的。
 *
 * @author He Han
 */
public class CustomDictionary
{
    /**
     * 默认实例
     */
    public static DynamicCustomDictionary DEFAULT = new DynamicCustomDictionary(HanLP.Config.CustomDictionaryPath);

    /**
     * 加载词典
     *
     * @param mainPath 缓存文件文件名
     * @param path     自定义词典
     * @param isCache  是否缓存结果
     */
    public static boolean loadMainDictionary(String mainPath, String path[], DoubleArrayTrie<CoreDictionary.Attribute> dat, boolean isCache)
    {
        return DynamicCustomDictionary.loadMainDictionary(mainPath, path, dat, isCache);
    }


    /**
     * 加载用户词典（追加）
     *
     * @param path                  词典路径
     * @param defaultNature         默认词性
     * @param customNatureCollector 收集用户词性
     * @return
     */
    public static boolean load(String path, Nature defaultNature, TreeMap<String, CoreDictionary.Attribute> map, LinkedHashSet<Nature> customNatureCollector)
    {
        return DynamicCustomDictionary.load(path, defaultNature, map, customNatureCollector);
    }


    /**
     * 往自定义词典中插入一个新词（非覆盖模式）<br>
     * 动态增删不会持久化到词典文件
     *
     * @param word                新词 如“裸婚”
     * @param natureWithFrequency 词性和其对应的频次，比如“nz 1 v 2”，null时表示“nz 1”
     * @return 是否插入成功（失败的原因可能是不覆盖、natureWithFrequency有问题等，后者可以通过调试模式了解原因）
     */
    public static boolean add(String word, String natureWithFrequency)
    {
        return DEFAULT.add(word, natureWithFrequency);
    }

    /**
     * 往自定义词典中插入一个新词（非覆盖模式）<br>
     * 动态增删不会持久化到词典文件
     *
     * @param word 新词 如“裸婚”
     * @return 是否插入成功（失败的原因可能是不覆盖等，可以通过调试模式了解原因）
     */
    public static boolean add(String word)
    {
        return DEFAULT.add(word);
    }

    /**
     * 往自定义词典中插入一个新词（覆盖模式）<br>
     * 动态增删不会持久化到词典文件
     *
     * @param word                新词 如“裸婚”
     * @param natureWithFrequency 词性和其对应的频次，比如“nz 1 v 2”，null时表示“nz 1”。
     * @return 是否插入成功（失败的原因可能是natureWithFrequency问题，可以通过调试模式了解原因）
     */
    public static boolean insert(String word, String natureWithFrequency)
    {
        return DEFAULT.insert(word, natureWithFrequency);
    }

    /**
     * 以覆盖模式增加新词<br>
     * 动态增删不会持久化到词典文件
     *
     * @param word
     * @return
     */
    public static boolean insert(String word)
    {
        return DEFAULT.insert(word);
    }

    public static boolean loadDat(String path, DoubleArrayTrie<CoreDictionary.Attribute> dat)
    {
        return DynamicCustomDictionary.loadDat(path, HanLP.Config.CustomDictionaryPath, dat);
    }

    /**
     * 从磁盘加载双数组
     *
     * @param path          主词典路径
     * @param customDicPath 用户词典路径
     * @return
     */
    public static boolean loadDat(String path, String customDicPath[], DoubleArrayTrie<CoreDictionary.Attribute> dat)
    {
        return DynamicCustomDictionary.loadDat(path, customDicPath, dat);
    }

    /**
     * 查单词
     *
     * @param key
     * @return
     */
    public static CoreDictionary.Attribute get(String key)
    {
        return DEFAULT.get(key);
    }

    /**
     * 删除单词<br>
     * 动态增删不会持久化到词典文件
     *
     * @param key
     */
    public static void remove(String key)
    {
        DEFAULT.remove(key);
    }

    /**
     * 前缀查询
     *
     * @param key
     * @return
     */
    public static LinkedList<Map.Entry<String, CoreDictionary.Attribute>> commonPrefixSearch(String key)
    {
        return DEFAULT.commonPrefixSearch(key);
    }

    /**
     * 前缀查询
     *
     * @param chars
     * @param begin
     * @return
     */
    public static LinkedList<Map.Entry<String, CoreDictionary.Attribute>> commonPrefixSearch(char[] chars, int begin)
    {
        return DEFAULT.commonPrefixSearch(chars, begin);
    }

    public static BaseSearcher getSearcher(String text)
    {
        return new Searcher(text);
    }

    @Override
    public String toString()
    {
        return "CustomDictionary{" +
            "trie=" + DEFAULT.trie +
            '}';
    }

    /**
     * 词典中是否含有词语
     *
     * @param key 词语
     * @return 是否包含
     */
    public static boolean contains(String key)
    {
        return DEFAULT.contains(key);
    }

    /**
     * 获取一个BinTrie的查询工具
     *
     * @param charArray 文本
     * @return 查询者
     */
    public static BaseSearcher getSearcher(char[] charArray)
    {
        return new Searcher(charArray);
    }

    static class Searcher extends BaseSearcher<CoreDictionary.Attribute>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        private LinkedList<Map.Entry<String, CoreDictionary.Attribute>> entryList;

        protected Searcher(char[] c)
        {
            super(c);
            entryList = new LinkedList<Map.Entry<String, CoreDictionary.Attribute>>();
        }

        protected Searcher(String text)
        {
            super(text);
            entryList = new LinkedList<Map.Entry<String, CoreDictionary.Attribute>>();
        }

        @Override
        public Map.Entry<String, CoreDictionary.Attribute> next()
        {
            // 保证首次调用找到一个词语
            while (entryList.size() == 0 && begin < c.length)
            {
                entryList = DEFAULT.trie.commonPrefixSearchWithValue(c, begin);
                ++begin;
            }
            // 之后调用仅在缓存用完的时候调用一次
            if (entryList.size() == 0 && begin < c.length)
            {
                entryList = DEFAULT.trie.commonPrefixSearchWithValue(c, begin);
                ++begin;
            }
            if (entryList.size() == 0)
            {
                return null;
            }
            Map.Entry<String, CoreDictionary.Attribute> result = entryList.getFirst();
            entryList.removeFirst();
            offset = begin - 1;
            return result;
        }
    }

    /**
     * 获取词典对应的trie树
     *
     * @return
     * @deprecated 谨慎操作，有可能废弃此接口
     */
    public static BinTrie<CoreDictionary.Attribute> getTrie()
    {
        return DEFAULT.getTrie();
    }

    /**
     * 解析一段文本（目前采用了BinTrie+DAT的混合储存形式，此方法可以统一两个数据结构）
     *
     * @param text      文本
     * @param processor 处理器
     */
    public static void parseText(char[] text, AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute> processor)
    {
        DEFAULT.parseText(text, processor);
    }

    /**
     * 解析一段文本（目前采用了BinTrie+DAT的混合储存形式，此方法可以统一两个数据结构）
     *
     * @param text      文本
     * @param processor 处理器
     */
    public static void parseText(String text, AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute> processor)
    {
        DEFAULT.parseText(text, processor);
    }

    /**
     * 最长匹配
     *
     * @param text      文本
     * @param processor 处理器
     */
    public static void parseLongestText(String text, AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute> processor)
    {
        DEFAULT.parseLongestText(text, processor);
    }

    /**
     * 热更新（重新加载）<br>
     * 集群环境（或其他IOAdapter）需要自行删除缓存文件（路径 = HanLP.Config.CustomDictionaryPath[0] + Predefine.BIN_EXT）
     *
     * @return 是否加载成功
     */
    public static boolean reload()
    {
        return DEFAULT.reload();
    }
}
