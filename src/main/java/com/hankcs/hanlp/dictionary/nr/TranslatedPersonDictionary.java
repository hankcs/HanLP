/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/12 14:45</create-date>
 *
 * <copyright file="TranslatedPersonDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.nr;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.Predefine;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.TreeMap;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 翻译人名词典，储存和识别翻译人名
 * @author hankcs
 */
public class TranslatedPersonDictionary
{
    static String path = HanLP.Config.TranslatedPersonDictionaryPath;
    static DoubleArrayTrie<Boolean> trie;

    static
    {
        long start = System.currentTimeMillis();
        if (!load())
        {
            throw new IllegalArgumentException("音译人名词典" + path + "加载失败");
        }

        logger.info("音译人名词典" + path + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
    }

    static boolean load()
    {
        trie = new DoubleArrayTrie<Boolean>();
        if (loadDat()) return true;
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            String line;
            TreeMap<String, Boolean> map = new TreeMap<String, Boolean>();
            TreeMap<Character, Integer> charFrequencyMap = new TreeMap<Character, Integer>();
            while ((line = br.readLine()) != null)
            {
                map.put(line, true);
                // 音译人名常用字词典自动生成
                for (char c : line.toCharArray())
                {
                    // 排除一些过于常用的字
                    if ("不赞".indexOf(c) >= 0) continue;
                    Integer f = charFrequencyMap.get(c);
                    if (f == null) f = 0;
                    charFrequencyMap.put(c, f + 1);
                }
            }
            br.close();
            map.put(String.valueOf('·'), true);
//            map.put(String.valueOf('-'), true);
//            map.put(String.valueOf('—'), true);
            // 将常用字也加进去
            for (Map.Entry<Character, Integer> entry : charFrequencyMap.entrySet())
            {
                if (entry.getValue() < 10) continue;
                map.put(String.valueOf(entry.getKey()), true);
            }
            logger.info("音译人名词典" + path + "开始构建双数组……");
            trie.build(map);
            logger.info("音译人名词典" + path + "开始编译DAT文件……");
            logger.info("音译人名词典" + path + "编译结果：" + saveDat());
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
     * @return
     */
    static boolean saveDat()
    {
        return trie.save(path + Predefine.TRIE_EXT);
    }

    static boolean loadDat()
    {
        return trie.load(path + Predefine.TRIE_EXT);
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
     * 时报包含key，且key至少长length
     * @param key
     * @param length
     * @return
     */
    public static boolean containsKey(String key, int length)
    {
        if (!trie.containsKey(key)) return false;
        return key.length() >= length;
    }
}
