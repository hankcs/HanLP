/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/12 20:17</create-date>
 *
 * <copyright file="JapanesePersonDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.nr;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.utility.Predefine;

import java.io.*;
import java.util.LinkedList;
import java.util.Map;
import java.util.TreeMap;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class JapanesePersonDictionary
{
    static String path = HanLP.Config.JapanesePersonDictionaryPath;
    static DoubleArrayTrie<Character> trie;
    /**
     * 姓
     */
    public static final char X = 'x';
    /**
     * 名
     */
    public static final char M = 'm';
    /**
     * bad case
     */
    public static final char A = 'A';

    static
    {
        long start = System.currentTimeMillis();
        if (!load())
        {
            throw new IllegalArgumentException("日本人名词典" + path + "加载失败");
        }

        logger.info("日本人名词典" + HanLP.Config.PinyinDictionaryPath + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
    }

    static boolean load()
    {
        trie = new DoubleArrayTrie<Character>();
        if (loadDat()) return true;
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            String line;
            TreeMap<String, Character> map = new TreeMap<String, Character>();
            while ((line = br.readLine()) != null)
            {
                String[] param = line.split(" ", 2);
                map.put(param[0], param[1].charAt(0));
            }
            br.close();
            logger.info("日本人名词典" + path + "开始构建双数组……");
            trie.build(map);
            logger.info("日本人名词典" + path + "开始编译DAT文件……");
            logger.info("日本人名词典" + path + "编译结果：" + saveDat(map));
        }
        catch (Exception e)
        {
            logger.severe("自定义词典" + path + "读取错误！" + e);
            return false;
        }

        return true;
    }

    /**
     * 保存dat到磁盘
     * @param map
     * @return
     */
    static boolean saveDat(TreeMap<String, Character> map)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(IOUtil.newOutputStream(path + Predefine.VALUE_EXT)));
            out.writeInt(map.size());
            for (Character character : map.values())
            {
                out.writeChar(character);
            }
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("保存值" + path + Predefine.VALUE_EXT + "失败" + e);
            return false;
        }
        return trie.save(path + Predefine.TRIE_EXT);
    }

    static boolean loadDat()
    {
        ByteArray byteArray = ByteArray.createByteArray(path + Predefine.VALUE_EXT);
        if (byteArray == null) return false;
        int size = byteArray.nextInt();
        Character[] valueArray = new Character[size];
        for (int i = 0; i < valueArray.length; ++i)
        {
            valueArray[i] = byteArray.nextChar();
        }
        return trie.load(path + Predefine.TRIE_EXT, valueArray);
    }

    /**
     * 是否包含key
     * @param key
     * @return
     */
    public static boolean containsKey(String key)
    {
        return trie.containsKey(key);
    }

    /**
     * 包含key，且key至少长length
     * @param key
     * @param length
     * @return
     */
    public static boolean containsKey(String key, int length)
    {
        if (!trie.containsKey(key)) return false;
        return key.length() >= length;
    }

    public static Character get(String key)
    {
        return trie.get(key);
    }

    public static DoubleArrayTrie<Character>.LongestSearcher getSearcher(char[] charArray)
    {
        return trie.getLongestSearcher(charArray, 0);
    }

    /**
     * 最长分词
     */
    public static class Searcher extends BaseSearcher<Character>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        DoubleArrayTrie<Character> trie;

        protected Searcher(char[] c, DoubleArrayTrie<Character> trie)
        {
            super(c);
            this.trie = trie;
        }

        protected Searcher(String text, DoubleArrayTrie<Character> trie)
        {
            super(text);
            this.trie = trie;
        }

        @Override
        public Map.Entry<String, Character> next()
        {
            // 保证首次调用找到一个词语
            Map.Entry<String, Character> result = null;
            while (begin < c.length)
            {
                LinkedList<Map.Entry<String, Character>> entryList = trie.commonPrefixSearchWithValue(c, begin);
                if (entryList.size() == 0)
                {
                    ++begin;
                }
                else
                {
                    result = entryList.getLast();
                    offset = begin;
                    begin += result.getKey().length();
                    break;
                }
            }
            if (result == null)
            {
                return null;
            }
            return result;
        }
    }
}
