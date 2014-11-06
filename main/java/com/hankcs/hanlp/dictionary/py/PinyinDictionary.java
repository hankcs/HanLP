/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/2 12:41</create-date>
 *
 * <copyright file="PinyinDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.py;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.utility.Predefine;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class PinyinDictionary
{
    static DoubleArrayTrie<Pinyin[]> trie = new DoubleArrayTrie<>();
    public static final Pinyin[] pinyins = Pinyin.values();

    static
    {
        long start = System.currentTimeMillis();
        if (!load(HanLP.Config.PinyinDictionaryPath))
        {
            throw new IllegalArgumentException("拼音词典" + HanLP.Config.PinyinDictionaryPath + "加载失败");
        }

        logger.info("拼音词典" + HanLP.Config.PinyinDictionaryPath + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
    }

    /**
     * 读取词典
     * @param path
     * @return
     */
    static boolean load(String path)
    {
        String datPath = path;
        if (loadDat(datPath)) return true;
        // 从文本中载入并且尝试生成dat
        StringDictionary dictionary = new StringDictionary("=");
        if (!dictionary.load(path)) return false;
        TreeMap<String, Pinyin[]> map = new TreeMap<>();
        for (Map.Entry<String, String> entry : dictionary.entrySet())
        {
            String[] args = entry.getValue().split(",");
            Pinyin[] pinyinValue = new Pinyin[args.length];
            for (int i = 0; i < pinyinValue.length; ++i)
            {
                try
                {
                    Pinyin pinyin = Pinyin.valueOf(args[i]);
                    pinyinValue[i] = pinyin;
                }
                catch (IllegalArgumentException e)
                {
                    logger.severe("读取拼音词典" + path + "失败，问题出在【" + entry + "】，异常是" + e);
                    return false;
                }
            }
            map.put(entry.getKey(), pinyinValue);
        }
        int resultCode = trie.build(map);
        if (resultCode < 0)
        {
            logger.warning(path + "构建DAT失败，错误码：" + resultCode);
            return false;
        }
        logger.info("正在缓存双数组" + datPath);
        saveDat(datPath, trie, map.entrySet());
        return true;
    }

    static boolean loadDat(String path)
    {
        ByteArray byteArray = ByteArray.createByteArray(path + Predefine.VALUE_EXT);
        if (byteArray == null) return false;
        int size = byteArray.nextInt();
        Pinyin[][] valueArray = new Pinyin[size][];
        for (int i = 0; i < valueArray.length; ++i)
        {
            int length = byteArray.nextInt();
            valueArray[i] = new Pinyin[length];
            for (int j = 0; j < length; ++j)
            {
                valueArray[i][j] = pinyins[byteArray.nextInt()];
            }
        }
        if (!trie.load(path + Predefine.TRIE_EXT, valueArray)) return false;
        return true;
    }

    static boolean saveDat(String path, DoubleArrayTrie<Pinyin[]> trie, Set<Map.Entry<String, Pinyin[]>> entrySet)
    {
        if (!trie.save(path + Predefine.TRIE_EXT)) return false;
        try
        {
            DataOutputStream out = new DataOutputStream(new FileOutputStream(path + Predefine.VALUE_EXT));
            out.writeInt(entrySet.size());
            for (Map.Entry<String, Pinyin[]> entry : entrySet)
            {
                Pinyin[] value = entry.getValue();
                out.writeInt(value.length);
                for (Pinyin pinyin : value)
                {
                    out.writeInt(pinyin.ordinal());
                }
            }
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("缓存值dat" + path + "失败");
            return false;
        }

        return true;
    }

    public static Pinyin[] get(String key)
    {
        return trie.get(key);
    }

    /**
     * 转为拼音
     * @param text
     * @return List形式的拼音，对应每一个字（所谓字，指的是任意字符）
     */
    public static List<Pinyin> convertToPinyin(String text)
    {
        return segLongest(text.toCharArray(), trie);
    }

    /**
     * 转为拼音
     * @param text
     * @return 数组形式的拼音
     */
    public static Pinyin[] convertToPinyinArray(String text)
    {
        return convertToPinyin(text).toArray(new Pinyin[0]);
    }

    public static BaseSearcher getSearcher(char[] charArray, DoubleArrayTrie<Pinyin[]> trie)
    {
        return new Searcher(charArray, trie);
    }

    protected static List<Pinyin> segLongest(char[] charArray, DoubleArrayTrie<Pinyin[]> trie)
    {
        List<Pinyin> pinyinList = new ArrayList<>(charArray.length);
        BaseSearcher searcher = getSearcher(charArray, trie);
        Map.Entry<String, Pinyin[]> entry;
        int p = 0;  // 当前处理到什么位置
        int offset;
        while ((entry = searcher.next()) != null)
        {
            offset = searcher.getOffset();
            // 补足没查到的词
            while (p < offset)
            {
                pinyinList.add(Pinyin.none5);
                ++p;
            }
            int length = entry.getKey().length();
            Pinyin[] value = entry.getValue();
            if (length == 1) pinyinList.add(value[0]);
            else
            {
                for (int i = 0; i < length; ++i)
                {
                    pinyinList.add(value[i]);
                }
            }
            p = offset + length;
        }
        // 补足没查到的词
        while (p < charArray.length)
        {
            pinyinList.add(Pinyin.none5);
            ++p;
        }
        return pinyinList;
    }

    public static class Searcher extends BaseSearcher<Pinyin[]>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        DoubleArrayTrie<Pinyin[]> trie;

        protected Searcher(char[] c, DoubleArrayTrie<Pinyin[]> trie)
        {
            super(c);
            this.trie = trie;
        }

        protected Searcher(String text, DoubleArrayTrie<Pinyin[]> trie)
        {
            super(text);
            this.trie = trie;
        }

        @Override
        public Map.Entry<String, Pinyin[]> next()
        {
            // 保证首次调用找到一个词语
            Map.Entry<String, Pinyin[]> result = null;
            while (begin < c.length)
            {
                LinkedList<Map.Entry<String, Pinyin[]>> entryList = trie.commonPrefixSearchWithValue(c, begin);
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
