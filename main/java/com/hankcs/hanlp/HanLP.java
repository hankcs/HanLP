/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/17 19:02</create-date>
 *
 * <copyright file="HanLP.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp;

import java.util.Properties;

/**
 * 常用接口静态化
 * @author hankcs
 */
public class HanLP
{
    /**
     * 库的配置
     */
    public static class Config
    {
        /**
         * 开发模式
         */
        public final static boolean DEBUG = false;
        /**
         * 核心词典路径
         */
        public static String CoreDictionaryPath = "data/dictionary/CoreNatureDictionary.txt";
        /**
         * 2元语法词典路径
         */
        public static String BiGramDictionaryPath = "data/dictionary/CoreNatureDictionary.ngram.txt";
        /**
         * 停用词词典路径
         */
        public static String CoreStopWordDictionaryPath = "data/dictionary/stopwords.txt";
        /**
         * 同义词词典路径
         */
        public static String CoreSynonymDictionaryDictionaryPath = "data/dictionary/synonym/CoreSynonym.txt";
        /**
         * 人名词典路径
         */
        public static String PersonDictionaryPath = "data/dictionary/person/combined.txt";
        /**
         * 人名词典转移矩阵路径
         */
        public static String PersonDictionaryTrPath = "data/dictionary/person/nr.tr.txt";

        static
        {
            // 自动读取配置
            Properties p = new Properties();
            try
            {
                p.load(Thread.currentThread().getContextClassLoader().getResourceAsStream("HanLP.properties"));
                String root = p.getProperty("root");
                CoreDictionaryPath = root + p.getProperty("CoreDictionaryPath");
                BiGramDictionaryPath = root + p.getProperty("BiGramDictionaryPath");
                CoreStopWordDictionaryPath = root + p.getProperty("CoreStopWordDictionaryPath");
                CoreSynonymDictionaryDictionaryPath = root + p.getProperty("CoreSynonymDictionaryDictionaryPath");
                PersonDictionaryPath = root + p.getProperty("PersonDictionaryPath");
                PersonDictionaryTrPath = root + p.getProperty("PersonDictionaryTrPath");
            }
            catch (Exception e)
            {
//                e.printStackTrace();
                if (DEBUG)
                {
                    System.out.println("没有找到HanLP.properties，将采用默认配置");
                }
            }
        }
    }
}
