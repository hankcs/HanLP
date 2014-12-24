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
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.utility.Predefine;

import java.io.*;
import java.util.logging.Level;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 核心词典的二元接续词典，采用整型储存，高性能
 *
 * @author hankcs
 */
public class CoreBiGramTableDictionary
{
    static int[][][] table;
    public final static String path = HanLP.Config.BiGramDictionaryPath;
    final static String datPath = HanLP.Config.BiGramDictionaryPath + ".table" + Predefine.BIN_EXT;

    static
    {
        logger.info("开始加载二元词典" + path + ".table");
        long start = System.currentTimeMillis();
        if (!load(path))
        {
            logger.severe("二元词典加载失败");
            System.exit(-1);
        }
        else
        {
            logger.info(path + ".table" + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        }
    }

    static boolean load(String path)
    {
        if (loadDat(datPath)) return true;
        table = new int[CoreDictionary.trie.size()][][];
        BufferedReader br;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] params = line.split("\\s");
                String[] twoWord = params[0].split("@", 2);
                String a = twoWord[0];
                int idA = CoreDictionary.trie.exactMatchSearch(a);
                if (idA == -1)
                {
//                    if (HanLP.Config.DEBUG)
//                        logger.warning(line + " 中的 " + a + "不存在于核心词典，将会忽略这一行");
                    continue;
                }
                String b = twoWord[1];
                int idB = CoreDictionary.trie.exactMatchSearch(b);
                if (idB == -1)
                {
//                    if (HanLP.Config.DEBUG)
//                        logger.warning(line + " 中的 " + b + "不存在于核心词典，将会忽略这一行");
                    continue;
                }
                int freq = Integer.parseInt(params[1]);
                if (table[idA] == null)
                {
                    table[idA] = new int[1][2];
                    table[idA][0][0] = idB;
                    table[idA][0][1] = freq;
                }
                else
                {
                    int[][] newLine = new int[table[idA].length + 1][2];
                    int index = binarySearch(table[idA], idB);
                    int insert = -(index + 1);
                    System.arraycopy(table[idA], 0, newLine, 0, insert);
                    System.arraycopy(table[idA], insert, newLine, insert + 1, table[idA].length - insert);
                    newLine[insert][0] = idB;
                    newLine[insert][1] = freq;
                    table[idA] = newLine;
                }
            }
            br.close();
            logger.info("二元词典读取完毕:" + path + "，开始构建双数组Trie树(DoubleArrayTrie)……");
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
        if (!saveDat(datPath))
        {
            logger.warning("缓存二元词典到" + datPath + "失败");
        }
        return true;
    }

    static boolean saveDat(String path)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(new FileOutputStream(path));
            out.writeInt(table.length);
            for (int[][] line : table)
            {
                if (line == null)
                {
                    out.writeInt(0);
                    continue;
                }
                out.writeInt(line.length);
                for (int[] row : line)
                {
//                    out.writeInt(row.length);// 固定为2
                    out.writeInt(row[0]);
                    out.writeInt(row[1]);
                }
            }
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
        ByteArray byteArray = ByteArray.createByteArray(path);
        if (byteArray == null) return false;

        int length = byteArray.nextInt();
        table = new int[length][][];
        for (int i = 0; i < table.length; ++i)
        {
            length = byteArray.nextInt();
            if (length == 0) continue;
            table[i] = new int[length][2];
            for (int j = 0; j < length; ++j)
            {
                table[i][j][0] = byteArray.nextInt();
                table[i][j][1] = byteArray.nextInt();
            }
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

    /**
     * 获取共现频次
     *
     * @param a 第一个词
     * @param b 第二个词
     * @return 第一个词@第二个词出现的频次
     */
    public static int getBiFrequency(String a, String b)
    {
        int idA = CoreDictionary.trie.exactMatchSearch(a);
        if (idA == -1)
        {
            return 0;
        }
        int idB = CoreDictionary.trie.exactMatchSearch(b);
        if (idB == -1)
        {
            return 0;
        }
        int index = binarySearch(table[idA], idB);
        if (index < 0) return 0;
        return table[idA][index][1];
    }

    public static int getBiFrequency(int idA, int idB)
    {
        if (idA == -1) return 0;
        if (table[idA] == null) return 0;
        if (idB == -1) return 0;
        int index = binarySearch(table[idA], idB);
        if (index < 0) return 0;
        return table[idA][index][1];
    }

//    public static int getBiFrequency(Vertex from, Vertex to)
//    {
//    }

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
