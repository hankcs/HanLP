/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 13:13</create-date>
 *
 * <copyright file="CommonSynonymDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.common;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.synonym.Synonym;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.TreeMap;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 一个没有指定资源位置的通用同义词词典
 *
 * @author hankcs
 */
public class CommonSynonymDictionary
{
    DoubleArrayTrie<SynonymItem> trie;

    private CommonSynonymDictionary()
    {
    }

    public static CommonSynonymDictionary create(InputStream inputStream)
    {
        CommonSynonymDictionary dictionary = new CommonSynonymDictionary();
        if (dictionary.load(inputStream))
        {
            return dictionary;
        }

        return null;
    }

    public boolean load(InputStream inputStream)
    {
        trie = new DoubleArrayTrie<SynonymItem>();
        TreeMap<String, SynonymItem> treeMap = new TreeMap<String, SynonymItem>();
        String line = null;
        try
        {
            BufferedReader bw = new BufferedReader(new InputStreamReader(inputStream));
            while ((line = bw.readLine()) != null)
            {
                String[] args = line.split(" ");
                List<Synonym> synonymList = Synonym.create(args);
                char type = args[0].charAt(args[0].length() - 1);
                for (Synonym synonym : synonymList)
                {
                    treeMap.put(synonym.realWord, new SynonymItem(synonym, synonymList, type));
                    // 这里稍微做个test
                    //assert synonym.getIdString().startsWith(line.split(" ")[0].substring(0, line.split(" ")[0].length() - 1)) : "词典有问题" + line + synonym.toString();
                }
            }
            bw.close();
            int resultCode = trie.build(treeMap);
            if (resultCode != 0)
            {
                logger.warning("构建" + inputStream + "失败，错误码" + resultCode);
                return false;
            }
        }
        catch (Exception e)
        {
            logger.warning("读取" + inputStream + "失败，可能由行" + line + "造成");
            return false;
        }
        return true;
    }

    public SynonymItem get(String key)
    {
        return trie.get(key);
    }

    /**
     * 语义距离
     *
     * @param a
     * @param b
     * @return
     */
    public long distance(String a, String b)
    {
        SynonymItem itemA = get(a);
        if (itemA == null) return Long.MAX_VALUE / 3;
        SynonymItem itemB = get(b);
        if (itemB == null) return Long.MAX_VALUE / 3;

        return itemA.distance(itemB);
    }

    /**
     * 词典中的一个条目
     */
    public static class SynonymItem
    {
        /**
         * 条目的key
         */
        public Synonym entry;
        /**
         * 条目的value，是key的同义词列表
         */
        public List<Synonym> synonymList;

        public static enum Type
        {
            /**
             * 完全同义词，对应词典中的=号
             */
            EQUAL,
            /**
             * 同类词，对应#
             */
            LIKE,
            /**
             * 封闭词，没有同义词或同类词
             */
            SINGLE,

            /**
             * 未定义，通常属于非词典中的词
             */
            UNDEFINED,
        }

        /**
         * 这个条目的类型，同义词或同类词或封闭词
         */
        public Type type;

        public SynonymItem(Synonym entry, List<Synonym> synonymList, Type type)
        {
            this.entry = entry;
            this.synonymList = synonymList;
            this.type = type;
        }

        public SynonymItem(Synonym entry, List<Synonym> synonymList, char type)
        {
            this.entry = entry;
            this.synonymList = synonymList;
            switch (type)
            {
                case '=':
                    this.type = Type.EQUAL;
                    break;
                case '#':
                    this.type = Type.LIKE;
                    break;
                default:
                    this.type = Type.SINGLE;
                    break;
            }
        }

        @Override
        public String toString()
        {
            final StringBuilder sb = new StringBuilder();
            sb.append(entry);
            sb.append(' ');
            sb.append(type);
            sb.append(' ');
            sb.append(synonymList);
            return sb.toString();
        }

        /**
         * 语义距离
         *
         * @param other
         * @return
         */
        public long distance(SynonymItem other)
        {
            return entry.distance(other.entry);
        }

        /**
         * 创建一个@类型的词典之外的条目
         *
         * @param word
         * @return
         */
        public static SynonymItem createUndefined(String word)
        {
            SynonymItem item = new SynonymItem(new Synonym(word, word.hashCode() * 1000000 + Long.MAX_VALUE / 3), null, Type.UNDEFINED);
            return item;
        }
    }
}
