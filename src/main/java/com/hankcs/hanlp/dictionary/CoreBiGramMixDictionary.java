/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/24 12:46</create-date>
 *
 * <copyright file="CoreBiGramDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.utility.ByteUtil;
import com.hankcs.hanlp.utility.Predefine;

import java.io.*;
import java.util.Collection;
import java.util.TreeMap;
import java.util.logging.Level;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 核心词典的二元接续词典，混合采用词ID和词本身储存
 *
 * @author hankcs
 */
public class CoreBiGramMixDictionary
{
    static DoubleArrayTrie<Integer> trie;
    public final static String path = HanLP.Config.BiGramDictionaryPath;
    final static String datPath = HanLP.Config.BiGramDictionaryPath + ".mix" + Predefine.BIN_EXT;

    static
    {
        logger.info("开始加载二元词典" + path + ".mix");
        long start = System.currentTimeMillis();
        if (!load(path))
        {
            throw new IllegalArgumentException("二元词典加载失败");
        }
        else
        {
            logger.info(path + ".mix" + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        }
    }

    static boolean load(String path)
    {
        trie = new DoubleArrayTrie<Integer>();
        if (loadDat(datPath)) return true;
        TreeMap<String, Integer> map = new TreeMap<String, Integer>();
        BufferedReader br;
        try
        {
            br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            String line;
            StringBuilder sb = new StringBuilder();
            while ((line = br.readLine()) != null)
            {
                String[] params = line.split("\\s");
                String[] twoWord = params[0].split("@", 2);
                buildID(twoWord[0], sb);
                sb.append('@');
                buildID(twoWord[1], sb);
                int freq = Integer.parseInt(params[1]);
                map.put(sb.toString(), freq);
                sb.setLength(0);
            }
            br.close();
            logger.info("二元词典读取完毕:" + path + "，开始构建双数组Trie树(DoubleArrayTrie)……");
            trie.build(map);
        }
        catch (FileNotFoundException e)
        {
            logger.severe("二元词典" + path + "不存在！" + e);
            return false;
        }
        catch (IOException e)
        {
            logger.severe("二元词典" + path + "读取错误！" + e);
            return false;
        }
        logger.info("开始缓存二元词典到" + datPath);
        if (!saveDat(datPath, map))
        {
            logger.warning("缓存二元词典到" + datPath + "失败");
        }
        return true;
    }

    static boolean saveDat(String path, TreeMap<String, Integer> map)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(IOUtil.newOutputStream(path)));
            Collection<Integer> freqList = map.values();
            out.writeInt(freqList.size());
            for (int freq : freqList)
            {
                out.writeInt(freq);
            }
            trie.save(out);
            out.close();
        }
        catch (Exception e)
        {
            logger.log(Level.WARNING, "在缓存" + path + "时发生异常", e);
            return false;
        }

        return true;
    }

    static boolean loadDat(String path)
    {
        try
        {
            ByteArray byteArray = ByteArray.createByteArray(path);
            if (byteArray == null) return false;

            int size = byteArray.nextInt();
            Integer[] value = new Integer[size];
            for (int i = 0; i < size; i++)
            {
                value[i] = byteArray.nextInt();
            }
            if (!trie.load(byteArray, value)) return false;
        }
        catch (Exception e)
        {
            return false;
        }

        return true;
    }

    /**
     * 二分搜索
     *
     * @param a
     * @param key
     * @return
     */
    static int binarySearch(int[][] a, int key)
    {
        int low = 0;
        int high = a.length - 1;

        while (low <= high)
        {
            int mid = (low + high) >>> 1;
            int midVal = a[mid][0];

            if (midVal < key)
                low = mid + 1;
            else if (midVal > key)
                high = mid - 1;
            else
                return mid; // key found
        }
        return -(low + 1);  // key not found.
    }



//    public static int getBiFrequency(Vertex from, Vertex to)
//    {
//        StringBuilder key = new StringBuilder();
//        int idA = from.wordID;
//        if (idA == -1)
//        {
//            key.append(from.word);
//        }
//        else
//        {
//            key.append(ByteUtil.convertIntToTwoChar(idA));
//        }
//        key.append('@');
//        int idB = to.wordID;
//        if (idB == -1)
//        {
//            key.append(to.word);
//        }
//        else
//        {
//            key.append(ByteUtil.convertIntToTwoChar(idB));
//        }
//
//        Integer freq = trie.get(key.toString());
//        if (freq == null) return 0;
//        return freq;
//    }

    static void buildID(String word, StringBuilder sbStorage)
    {
        int id = CoreDictionary.trie.exactMatchSearch(word);
        if (id == -1)
        {
            sbStorage.append(word);
        }
        else
        {
            char[] twoChar = ByteUtil.convertIntToTwoChar(id);
            sbStorage.append(twoChar);
        }
    }

    /**
     * 获取词语的ID
     *
     * @param a
     * @return
     */
    public static int getWordID(String a)
    {
        return CoreDictionary.trie.exactMatchSearch(a);
    }
}
