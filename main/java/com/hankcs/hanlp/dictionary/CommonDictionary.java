/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 22:30</create-date>
 *
 * <copyright file="CommonDictioanry.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * 通用的词典，对应固定格式的词典，但是标签可以泛型化
 *
 * @author hankcs
 */
public abstract class CommonDictionary<V>
{
    static Logger logger = LoggerFactory.getLogger(CommonDictionary.class);
    DoubleArrayTrie<V> trie;

    public boolean load(String path)
    {
        trie = new DoubleArrayTrie<V>();
        V[] valueArray = onLoadValue(path);
        if (loadDat(path + ".trie.dat", valueArray)) return true;
        List<String> keyList = new ArrayList<>(valueArray.length);
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] paramArray = line.split("\\s");
                keyList.add(paramArray[0]);
            }
            br.close();
        }
        catch (Exception e)
        {
            logger.warn("读取{}失败", path, e);
        }
        int resultCode = trie.build(keyList, valueArray);
        if (resultCode != 0)
        {
            logger.warn("trie建立失败{},正在尝试排序后重载", resultCode);
            if (!sort(path))
            {
                return false;
            }
            load(path);
        }
        trie.save(path + ".trie.dat");
        logger.trace("{}加载成功", path);
        return true;
    }

    private boolean loadDat(String path, V[] valueArray)
    {
        if (trie.load(path, valueArray)) return true;
        return false;
    }

    /**
     * 查询一个单词
     *
     * @param key
     * @return 单词对应的条目
     */
    public V get(String key)
    {
        return trie.get(key);
    }

    /**
     * 是否含有键
     *
     * @param key
     * @return
     */
    public boolean contains(String key)
    {
        return get(key) != null;
    }

    /**
     * 词典大小
     * @return
     */
    public int size()
    {
        return trie.size();
    }

    /**
     * 排序这个词典
     * @param path
     * @return
     */
    public static boolean sort(String path)
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] argArray = line.split("\\s");
                map.put(argArray[0], line);
            }
            br.close();
            // 输出它们
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path)));
            for (Map.Entry<String, String> entry : map.entrySet())
            {
                bw.write(entry.getValue());
                bw.newLine();
            }
            bw.close();
        }
        catch (Exception e)
        {
            logger.warn("读取{}失败", path, e);
            return false;
        }
        return true;
    }

    /**
     * 实现此方法来加载值
     * @param path
     * @return
     */
    protected abstract V[] onLoadValue(String path);

    public BaseSearcher getSearcher(String text)
    {
        return new Searcher(text);
    }

    /**
     * 前缀搜索，长短都可匹配
     */
    public class Searcher extends BaseSearcher<V>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        private List<Map.Entry<String, V>> entryList;

        protected Searcher(char[] c)
        {
            super(c);
        }

        protected Searcher(String text)
        {
            super(text);
            entryList = new LinkedList<Map.Entry<String, V>>();
        }

        @Override
        public Map.Entry<String, V> next()
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
            Map.Entry<String, V> result = entryList.get(0);
            entryList.remove(0);
            offset = begin - 1;
            return result;
        }
    }
}
