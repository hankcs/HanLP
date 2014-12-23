/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/23 20:07</create-date>
 *
 * <copyright file="CoreDictionaryACDAT.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.Predefine;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 使用AhoCorasickDoubleArrayTrie实现的核心词典
 * @author hankcs
 */
public class CoreDictionaryACDAT
{
    public static AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> trie = new AhoCorasickDoubleArrayTrie<>();
    public final static String path = HanLP.Config.CoreDictionaryPath;
    public static final int totalFrequency = 221894;

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
        TreeMap<String, CoreDictionary.Attribute> map = new TreeMap<>();
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
                int natureCount = (param.length - 1) / 2;
                CoreDictionary.Attribute attribute = new CoreDictionary.Attribute(natureCount);
                for (int i = 0; i < natureCount; ++i)
                {
                    attribute.nature[i] = Enum.valueOf(Nature.class, param[1 + 2 * i]);
                    attribute.frequency[i] = Integer.parseInt(param[2 + 2 * i]);
                    attribute.totalFrequency += attribute.frequency[i];
                }
                map.put(param[0], attribute);
                MAX_FREQUENCY += attribute.totalFrequency;
            }
            logger.info("核心词典读入词条" + map.size() + " 全部频次" + MAX_FREQUENCY + "，耗时" + (System.currentTimeMillis() - start) + "ms");
            br.close();
            trie.build(map);
            logger.info("核心词典加载成功:" + trie.size() + "个词条");
            try
            {
                DataOutputStream out = new DataOutputStream(new FileOutputStream(path + Predefine.BIN_EXT));
                Collection<CoreDictionary.Attribute> attributeList = map.values();
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
                trie.save(out);
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
            ByteArray byteArray = ByteArray.createByteArray(path + Predefine.BIN_EXT);
            if (byteArray == null) return false;
            int size = byteArray.nextInt();
            CoreDictionary.Attribute[] attributes = new CoreDictionary.Attribute[size];
            final Nature[] natureIndexArray = Nature.values();
            for (int i = 0; i < size; ++i)
            {
                // 第一个是全部频次，第二个是词性个数
                int currentTotalFrequency = byteArray.nextInt();
                int length = byteArray.nextInt();
                attributes[i] = new CoreDictionary.Attribute(length);
                attributes[i].totalFrequency = currentTotalFrequency;
                for (int j = 0; j < length; ++j)
                {
                    attributes[i].nature[j] = natureIndexArray[byteArray.nextInt()];
                    attributes[i].frequency[j] = byteArray.nextInt();
                }
            }
            if (!trie.load(byteArray, attributes)) return false;
        }
        catch (Exception e)
        {
            logger.warning("读取失败，问题发生在" + e);
            return false;
        }
        return true;
    }
}
