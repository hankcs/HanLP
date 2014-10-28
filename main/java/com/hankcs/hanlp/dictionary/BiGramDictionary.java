/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/16 20:55</create-date>
 *
 * <copyright file="BiGramDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.utility.Utility;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 2元语法词典
 * @author hankcs
 */
public class BiGramDictionary
{
    static DoubleArrayTrie<Integer> trie;

    public final static String path = HanLP.Config.BiGramDictionaryPath;
    public static final int totalFrequency = 37545990;
    // 自动加载词典
    static
    {
        long start = System.currentTimeMillis();
        if (!load(path))
        {
            logger.severe("二元词典加载失败");
            System.exit(-1);
        }
        else
        {
            logger.info(path + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        }
    }
    public static boolean load(String path)
    {
        logger.info("二元词典开始加载:"+ path);
        trie = new DoubleArrayTrie<>();
        boolean create = !loadDat(path);
        if (!create) return true;
        List<String> wordList = new ArrayList<String>();
        List<Integer> freqList = new ArrayList<Integer>();
        BufferedReader br;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] params = line.split("\\s");
                String twoWord = params[0];
                int freq = Integer.parseInt(params[1]);
                wordList.add(twoWord);
                freqList.add(freq);
            }
            br.close();
            logger.info("二元词典读取完毕:" + path + "，开始构建DAT……");
        } catch (FileNotFoundException e)
        {
            logger.severe("二元词典" + path + "不存在！"+ e);
            return false;
        } catch (IOException e)
        {
            logger.severe("二元词典" + path + "读取错误！"+ e);
            return false;
        }

        int resultCode = trie.build(wordList, freqList);
        logger.info("二元词典DAT构建结果:{}"+ resultCode);
        if (resultCode < 0)
        {
            trie = new DoubleArrayTrie<Integer>();
            logger.info("二元词典排序中……");
            sortListForBuildTrie(wordList, freqList, path);
            logger.info("二元词典排序完毕，正在重试加载……");
            return load(path);
        }
        logger.info("二元词典加载成功:" + trie.size() + "个词条");
        // 试一试保存
        if (create)
        {
            // 先存双数组
            trie.save(path + ".trie.dat");
            // 后存值
            try
            {
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path + ".value.dat")));
                out.writeInt(freqList.size());
                for (int freq : freqList)
                {
                    out.writeInt(freq);
                }
                out.close();
            }
            catch (Exception e)
            {
                return false;
            }
        }
//        LogManager.getLogger().printf(Level.TRACE, "“·@自强”的频次:%d", trie.getValueAt(trie.exactMatchSearch("·@自强")));
        return true;
    }

    /**
     * 从dat文件中加载排好的trie
     * @param path
     * @return
     */
    private static boolean loadDat(String path)
    {
        try
        {
            long start = System.currentTimeMillis();
            FileInputStream fis = new FileInputStream(path + ".value.dat");
            FileChannel channel = fis.getChannel();
            int fileSize = (int) channel.size();
            ByteBuffer byteBuffer = ByteBuffer.allocate(fileSize);
            channel.read(byteBuffer);
            byteBuffer.flip();
            byte[] bytes = byteBuffer.array();
            byteBuffer.clear();
            channel.close();
            fis.close();

            int index = 0;
            int size = Utility.bytesHighFirstToInt(bytes, index);
            index += 4;
            Integer[] value = new Integer[size];
            for (int i = 0; i < size; i++)
            {
                value[i] = Utility.bytesHighFirstToInt(bytes, index);
                index += 4;
            }
            logger.info("加载值" + path + ".value.dat成功，耗时" + (System.currentTimeMillis() - start) + "ms");
            start = System.currentTimeMillis();
            if (!trie.load(path + ".trie.dat", value)) return false;
            logger.info("加载键" + path + ".trie.dat成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        }
        catch (Exception e)
        {
            return false;
        }

        return true;
    }

    /**
     * 找寻特殊字串，如未##串
     * @deprecated 没事就不要用了
     * @return 一个包含特殊词串的set
     */
    public static Set<String> _findSpecialString()
    {
        Set<String> stringSet = new HashSet<String>();
        BufferedReader br;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] params = line.split("\t");
                String twoWord = params[0];
                params = twoWord.split("@");
                for (String w : params)
                {
                    if (w.contains("##"))
                    {
                        stringSet.add(w);
                    }
                }
            }
            br.close();
        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        } catch (IOException e)
        {
            e.printStackTrace();
        }

        return stringSet;
    }

    /**
     * 获取共现频次
     * @param from 第一个词
     * @param to 第二个词
     * @return 第一个词@第二个词出现的频次
     */
    public static int getBiFrequency(String from, String to)
    {
        return getBiFrequency(from + '@' + to);
    }

    /**
     * 获取共现频次
     * @param twoWord 用@隔开的两个词
     * @return 共现频次
     */
    public static int getBiFrequency(String twoWord)
    {
        Integer result = trie.get(twoWord);
        return (result == null ? 0 : result);
    }

    /**
     * 接受键数组与值数组，排序以供建立trie树
     * @param wordList
     * @param freqList
     */
    private static void sortListForBuildTrie(List<String> wordList, List<Integer> freqList, String path)
    {
        BinTrie<Integer> binTrie = new BinTrie<Integer>();
        for (int i = 0; i < wordList.size(); ++i)
        {
            binTrie.put(wordList.get(i), freqList.get(i));
        }
        Collections.sort(wordList);
        try
        {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path)));
//            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "_sort.txt")));
            for (String w : wordList)
            {
                bw.write(w + '\t' + binTrie.get(w));
                bw.newLine();
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
}
