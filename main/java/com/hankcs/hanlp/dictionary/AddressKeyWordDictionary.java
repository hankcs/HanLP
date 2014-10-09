/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/07/2014/7/8 9:09</create-date>
 *
 * <copyright file="AddressDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class AddressKeyWordDictionary
{
    static Logger logger = LoggerFactory.getLogger(AddressKeyWordDictionary.class);
    static DoubleArrayTrie<Integer> trie = new DoubleArrayTrie<Integer>();
    public final static String path = "data/dictionary/address/keyword.txt";
    // 自动加载词典
    static
    {
        if (!load(path))
        {
            logger.error("地址词典加载失败");
            System.exit(-1);
        }
    }


    public static boolean load(String path)
    {
        logger.info("地址词典开始加载:{}", path);
        TreeMap<String, Integer> stringIntegerMap = new TreeMap<String, Integer>();
        BufferedReader br = null;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                String param[] = line.split(" ");
                stringIntegerMap.put(param[0], Integer.valueOf(param[1]));
            }
            logger.trace("地址词典读入词条{}", stringIntegerMap.size());
            br.close();
        } catch (FileNotFoundException e)
        {
            logger.error("地址词典" + path + "不存在！");
            e.printStackTrace();
            return false;
        } catch (IOException e)
        {
            logger.error("地址词典" + path + "读取错误！");
            e.printStackTrace();
            return false;
        }

        logger.trace("地址词典DAT构建结果:{}", trie.build(stringIntegerMap));
        logger.info("地址词典加载成功:{}个词条", trie.size());
        logger.trace("“市”对应的属性:{}", trie.getValueAt(trie.exactMatchSearch("市")).toString());
        return true;
    }

    public static int get(String key)
    {
        if (trie == null || key == null) return -1;
        Integer result = trie.get(key);
        if (result == null) return -1;
        return result;
    }
}
