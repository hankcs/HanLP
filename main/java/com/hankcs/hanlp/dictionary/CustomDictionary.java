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
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 用户自定义词典
 *
 * @author He Han
 */
public class CustomDictionary
{
    static BinTrie<CoreDictionary.Attribute> trie = new BinTrie<CoreDictionary.Attribute>();
    public final static String path = HanLP.Config.CustomDictionaryPath;

    // 自动加载词典
    static
    {
        long start = System.currentTimeMillis();
        if (!load(path))
        {
            logger.warning("自定义词典" + path + "加载失败");
        }
        else
        {
            logger.info("自定义词典加载成功:" + trie.size() + "个词条，耗时" + (System.currentTimeMillis() - start) + "ms");
        }
    }

    public static boolean load(String path)
    {
        logger.info("自定义词典开始加载:" + path);
        if (loadDat(path)) return true;
        List<String> wordList = new LinkedList<>();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                wordList.add(line);
                String[] param = line.split("\\s");
                int natureCount = (param.length - 1) / 2;
                CoreDictionary.Attribute attribute = new CoreDictionary.Attribute(natureCount);
                for (int i = 0; i < natureCount; ++i)
                {
                    attribute.nature[i] = Enum.valueOf(Nature.class, param[1 + 2 * i]);
                    attribute.frequency[i] = Integer.parseInt(param[2 + 2 * i]);
                    attribute.totalFrequency += attribute.frequency[i];
                }
                trie.put(param[0], attribute);
            }
            br.close();

            // 缓存成dat文件，下次加载会快很多
            if (!trie.save(path + ".trie.dat")) return false;
            // 缓存值文件
            List<CoreDictionary.Attribute> attributeList = new LinkedList<>();
            for (Map.Entry<String, CoreDictionary.Attribute> entry : trie.entrySet())
            {
                attributeList.add(entry.getValue());
            }
            DataOutputStream out = new DataOutputStream(new FileOutputStream(path + ".value.dat"));
            out.writeInt(attributeList.size());
            for (CoreDictionary.Attribute attribute : attributeList)
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
        catch (FileNotFoundException e)
        {
            logger.severe("自定义词典" + path + "不存在！" + e);
            return false;
        }
        catch (IOException e)
        {
            logger.severe("自定义词典" + path + "读取错误！" + e);
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
            CoreDictionary.Attribute[] attributes = new CoreDictionary.Attribute[size];
            final Nature[] natureIndexArray = Nature.values();
            for (int i = 0; i < size; ++i)
            {
                // 第一个是全部频次，第二个是词性个数
                int currentTotalFrequency = TextUtility.bytesHighFirstToInt(bytes, index);
                index += 4;
                int length = TextUtility.bytesHighFirstToInt(bytes, index);
                index += 4;
                attributes[i] = new CoreDictionary.Attribute(length);
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
            logger.info("二分数组" + path + ".trie.dat" + "加载完毕");
        }
        catch (Exception e)
        {
            logger.warning("读取失败，问题发生在" + e);
            return false;
        }
        return true;
    }

    /**
     * 查单词
     *
     * @param key
     * @return
     */
    public static CoreDictionary.Attribute get(String key)
    {
        return trie.get(key);
    }

    /**
     * 前缀查询
     *
     * @param key
     * @return
     */
    public static LinkedList<Map.Entry<String, CoreDictionary.Attribute>> commonPrefixSearch(String key)
    {
        return trie.commonPrefixSearchWithValue(key);
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
        return trie.commonPrefixSearchWithValue(chars, begin);
    }

    public static BaseSearcher getSearcher(String text)
    {
        return new Searcher(text);
    }

    @Override
    public String toString()
    {
        return "CustomDictionary{" +
                "trie=" + trie +
                '}';
    }

    static class Searcher extends BaseSearcher<CoreDictionary.Attribute>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        private List<Map.Entry<String, CoreDictionary.Attribute>> entryList;

        protected Searcher(char[] c)
        {
            super(c);
        }

        protected Searcher(String text)
        {
            super(text);
            entryList = new LinkedList<>();
        }

        @Override
        public Map.Entry<String, CoreDictionary.Attribute> next()
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
            Map.Entry<String, CoreDictionary.Attribute> result = entryList.get(0);
            entryList.remove(0);
            offset = begin - 1;
            return result;
        }
    }

    /**
     * 获取词典对应的trie树
     * @deprecated 谨慎操作，有可能废弃此接口
     * @return
     */
    public static BinTrie<CoreDictionary.Attribute> getTrie()
    {
        return trie;
    }
}
