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

import com.hankcs.hanlp.dictionary.ts.SimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalChineseDictionary;

import java.util.Properties;
import java.util.logging.Level;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 常用接口静态化
 *
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
        public static boolean DEBUG = false;
        /**
         * 核心词典路径
         */
        public static String CoreDictionaryPath = "data/dictionary/CoreNatureDictionary.txt";
        /**
         * 用户自定义词典路径
         */
        public static String CustomDictionaryPath = "data/dictionary/CustomDictionary.txt";
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
        /**
         * 繁简词典路径
         */
        public static String TraditionalChineseDictionaryPath = "data/dictionary/TraditionalChinese.txt";
        /**
         * 声母韵母语调词典
         */
        public static String SYTDictionaryPath = "data/dictionary/pinyin/SYTDictionary.txt";

        /**
         * 拼音词典路径
         */
        public static String PinyinDictionaryPath = "data/dictionary/pinyin/pinyin.txt";

        static
        {
            // 自动读取配置
            Properties p = new Properties()
            {
                @Override
                public String getProperty(String key)
                {
                    return super.getProperty(key, key);
                }
            };
            try
            {
                p.load(Thread.currentThread().getContextClassLoader().getResourceAsStream("HanLP.properties"));
                String root = p.getProperty("root", "");
                CoreDictionaryPath = root + p.getProperty("CoreDictionaryPath");
                BiGramDictionaryPath = root + p.getProperty("BiGramDictionaryPath");
                CoreStopWordDictionaryPath = root + p.getProperty("CoreStopWordDictionaryPath");
                CoreSynonymDictionaryDictionaryPath = root + p.getProperty("CoreSynonymDictionaryDictionaryPath");
                PersonDictionaryPath = root + p.getProperty("PersonDictionaryPath");
                PersonDictionaryTrPath = root + p.getProperty("PersonDictionaryTrPath");
                CustomDictionaryPath = root + p.getProperty("CustomDictionaryPath");
                TraditionalChineseDictionaryPath = root + p.getProperty("TraditionalChineseDictionaryPath");
                SYTDictionaryPath = root + p.getProperty("SYTDictionaryPath");
                PinyinDictionaryPath = root + p.getProperty("PinyinDictionaryPath");
            }
            catch (Exception e)
            {
//                if (!DEBUG)
//                    logger.warning("没有找到HanLP.properties，将采用默认配置");
            }
            if (!DEBUG)
            {
                logger.setLevel(Level.OFF);
            }
        }
        /**
         * 开启调试模式(会降低性能)
         */
        public static void enableDebug()
        {
            enableDebug(true);
        }

        public static void enableDebug(boolean enable)
        {
            DEBUG = enable;
            if (DEBUG)
            {
                logger.setLevel(Level.ALL);
            }
            else
            {
                logger.setLevel(Level.OFF);
            }
        }
    }

    /**
     * 简转繁
     * @param traditionalChineseString 繁体中文
     * @return 简体中文
     */
    public static String convertToSimplifiedChinese(String traditionalChineseString)
    {
        return TraditionalChineseDictionary.convertToSimplifiedChinese(traditionalChineseString);
    }

    /**
     * 繁转简
     * @param simplifiedChineseString 简体中文
     * @return 繁体中文
     */
    public static String convertToTraditionalChinese(String simplifiedChineseString)
    {
        return SimplifiedChineseDictionary.convertToTraditionalChinese(simplifiedChineseString);
    }

//    /**
//     * 开启调试模式
//     */
//    public static void enableDebug()
//    {
//        HanLP.Config.enableDebug();
//    }
}
