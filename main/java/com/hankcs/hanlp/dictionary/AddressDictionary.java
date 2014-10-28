/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/07/2014/7/8 14:23</create-date>
 *
 * <copyright file="AddressDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.recognition.ns.AddressType;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * 地址词典
 * @author hankcs
 */
public class AddressDictionary
{
    static DoubleArrayTrie<AddressType> trie = new DoubleArrayTrie<AddressType>();
    static
    {
        TreeMap<String, AddressType> storeMap = new TreeMap<String, AddressType>();
        load("data/dictionary/address/country.txt", AddressType.Country, storeMap);
        load("data/dictionary/address/province.txt", AddressType.Province, storeMap);
        load("data/dictionary/address/city.txt", AddressType.City, storeMap);
        load("data/dictionary/address/district.txt", AddressType.District, storeMap);
        load("data/dictionary/address/county.txt", AddressType.County, storeMap);
        load("data/dictionary/address/town.txt", AddressType.Town, storeMap);
        load("data/dictionary/address/street.txt", AddressType.Street, storeMap);
        load("data/dictionary/address/landmark.txt", AddressType.LandMark, storeMap);
        load("data/dictionary/address/SuffixBuildingUnit.txt", AddressType.SuffixBuildingUnit, storeMap);
        load("data/dictionary/address/SuffixDistrict.txt", AddressType.SuffixDistrict, storeMap);
        load("data/dictionary/address/SuffixLandMark.txt", AddressType.SuffixLandMark, storeMap);
        load("data/dictionary/address/relatedPos.txt", AddressType.RelatedPos, storeMap);
        load("data/dictionary/address/SuffixNumber.txt", AddressType.SuffixNumber, storeMap);
        load("data/dictionary/address/symbol.txt", AddressType.Symbol, storeMap);
//        load("data/dictionary/address/village.txt", AddressType.Village, storeMap);
        logger.info("trie构建结果：" + trie.build(storeMap));

    }
    static boolean load(String path, AddressType addressType, TreeMap<String, AddressType> storeMap)
    {
        try
        {
            logger.info("地址词典开始加载" + path);
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                // 逆序放入，分词的时候逆序最长匹配！
                storeMap.put(new StringBuilder(line).reverse().toString(), addressType);
            }
            br.close();
            return true;
        }
        catch (IOException e)
        {
            logger.info("地址词典加载" + path + "失败");
            e.printStackTrace();
        }

        return false;
    }

    /**
     * 找出一个字串最长匹配的尾部关键字
     * @param key
     * @return
     */
    public static AddressType commonSuffixSearch(String key)
    {
        BaseSearcher searcher = getSearcher(key);
        Map.Entry<String, AddressType> entry = searcher.next();
        if (entry != null)
        {
            return entry.getValue();
        }

        return null;
    }

    public static BaseSearcher getSearcher(String text)
    {
        return new Searcher(text);
    }

    public static class Searcher extends BaseSearcher<AddressType>
    {
        /**
         * 分词从何处开始，这是一个状态
         */
        int begin;

        private LinkedList<Map.Entry<String, AddressType>> entryList;

        protected Searcher(char[] c)
        {
            super(c);
        }

        protected Searcher(String text)
        {
            super(new StringBuilder(text).reverse().toString());
            entryList = new LinkedList<Map.Entry<String, AddressType>>();
        }

        @Override
        public Map.Entry<String, AddressType> next()
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
            Map.Entry<String, AddressType> result = entryList.getLast();    // 最长匹配
            result = new AbstractMap.SimpleEntry<String, AddressType>(new StringBuilder(result.getKey()).reverse().toString(), result.getValue());
            entryList.clear();
            begin = begin - 1 + result.getKey().length();
            return result;
        }

        @Override
        public int getOffset()
        {
            return c.length - begin;
        }
    }

}
