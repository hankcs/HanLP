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

import com.hankcs.hanlp.dictionary.py.Pinyin;
import com.hankcs.hanlp.dictionary.py.PinyinDictionary;
import com.hankcs.hanlp.dictionary.ts.SimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalChineseDictionary;
import com.hankcs.hanlp.seg.AbstractSegment;
import com.hankcs.hanlp.seg.Dijkstra.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandTokenizer;

import java.io.InputStreamReader;
import java.util.List;
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
         * 核心词典词性转移矩阵路径
         */
        public static String CoreDictionaryTransformMatrixDictionaryPath = "data/dictionary/CoreNatureDictionary.tr.txt";
        /**
         * 用户自定义词典路径
         */
        public static String CustomDictionaryPath[] = new String[]{"data/dictionary/custom/CustomDictionary.txt"};
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
         * 地名词典路径
         */
        public static String PlaceDictionaryPath = "data/dictionary/place/ns.txt";
        /**
         * 地名词典转移矩阵路径
         */
        public static String PlaceDictionaryTrPath = "data/dictionary/place/ns.tr.txt";
        /**
         * 地名词典路径
         */
        public static String OrganizationDictionaryPath = "data/dictionary/organization/nt.txt";
        /**
         * 地名词典转移矩阵路径
         */
        public static String OrganizationDictionaryTrPath = "data/dictionary/organization/nt.tr.txt";
        /**
         * 繁简词典路径
         */
        public static String TraditionalChineseDictionaryPath = "data/dictionary/tc/TraditionalChinese.txt";
        /**
         * 声母韵母语调词典
         */
        public static String SYTDictionaryPath = "data/dictionary/pinyin/SYTDictionary.txt";

        /**
         * 拼音词典路径
         */
        public static String PinyinDictionaryPath = "data/dictionary/pinyin/pinyin.txt";

        /**
         * 音译人名词典
         */
        public static String TranslatedPersonDictionaryPath = "data/dictionary/person/音译人名.txt";

        /**
         * 日本人名词典路径
         */
        public static String JapanesePersonDictionaryPath = "data/dictionary/person/日本人名词典.txt";

        /**
         * 词-词性-依存关系模型
         */
        public static String WordNatureModelPath = "data/model/dependency/WordNature.txt";

        /**
         * 最大熵-依存关系模型
         */
        public static String MaxEntModelPath = "data/model/dependency/MaxEntModel.txt";
        /**
         * 字符类型对应表
         */
        public static String CharTypePath = "data/dictionary/other/CharType.dat.yes";

        static
        {
            // 自动读取配置
            Properties p = new Properties();
            try
            {
                p.load(new InputStreamReader(Thread.currentThread().getContextClassLoader().getResourceAsStream("HanLP.properties"), "UTF-8"));
                String root = p.getProperty("root", "");
                CoreDictionaryPath = root + p.getProperty("CoreDictionaryPath", CoreDictionaryPath);
                BiGramDictionaryPath = root + p.getProperty("BiGramDictionaryPath", BiGramDictionaryPath);
                CoreStopWordDictionaryPath = root + p.getProperty("CoreStopWordDictionaryPath", CoreStopWordDictionaryPath);
                CoreSynonymDictionaryDictionaryPath = root + p.getProperty("CoreSynonymDictionaryDictionaryPath", CoreSynonymDictionaryDictionaryPath);
                PersonDictionaryPath = root + p.getProperty("PersonDictionaryPath", PersonDictionaryPath);
                PersonDictionaryTrPath = root + p.getProperty("PersonDictionaryTrPath", PersonDictionaryTrPath);
//                CustomDictionaryPath = root + p.getProperty("CustomDictionaryPath");
                String[] pathArray = p.getProperty("CustomDictionaryPath", "CustomDictionaryPath=dictionary/custom/CustomDictionary.txt").split(";");
                String prePath = root;
                for (int i = 0; i < pathArray.length; ++i)
                {
                    if (pathArray[i].startsWith(" "))
                    {
                        pathArray[i] = prePath + pathArray[i].trim();
                    }
                    else
                    {
                        pathArray[i] = root + pathArray[i];
                        int lastSplash = pathArray[i].lastIndexOf('/');
                        if (lastSplash != -1)
                        {
                            prePath = pathArray[i].substring(0, lastSplash + 1);
                        }
                    }
                }
                CustomDictionaryPath = pathArray;
                TraditionalChineseDictionaryPath = root + p.getProperty("TraditionalChineseDictionaryPath", TraditionalChineseDictionaryPath);
                SYTDictionaryPath = root + p.getProperty("SYTDictionaryPath", SYTDictionaryPath);
                PinyinDictionaryPath = root + p.getProperty("PinyinDictionaryPath", PinyinDictionaryPath);
                TranslatedPersonDictionaryPath = root + p.getProperty("TranslatedPersonDictionary", TranslatedPersonDictionaryPath);
                JapanesePersonDictionaryPath = root + p.getProperty("JapanesePersonDictionaryPath", JapanesePersonDictionaryPath);
                PlaceDictionaryPath = root + p.getProperty("PlaceDictionaryPath", PlaceDictionaryPath);
                PlaceDictionaryTrPath = root + p.getProperty("PlaceDictionaryTrPath", PlaceDictionaryTrPath);
                OrganizationDictionaryPath = root + p.getProperty("OrganizationDictionaryPath", OrganizationDictionaryPath);
                OrganizationDictionaryTrPath = root + p.getProperty("OrganizationDictionaryTrPath", OrganizationDictionaryTrPath);
                CharTypePath = root + p.getProperty("CharTypePath", CharTypePath);
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
     *
     * @param traditionalChineseString 繁体中文
     * @return 简体中文
     */
    public static String convertToSimplifiedChinese(String traditionalChineseString)
    {
        return TraditionalChineseDictionary.convertToSimplifiedChinese(traditionalChineseString);
    }

    /**
     * 繁转简
     *
     * @param simplifiedChineseString 简体中文
     * @return 繁体中文
     */
    public static String convertToTraditionalChinese(String simplifiedChineseString)
    {
        return SimplifiedChineseDictionary.convertToTraditionalChinese(simplifiedChineseString);
    }

    /**
     * 转化为拼音
     * @param text
     * @param separator
     * @param remainNone
     * @return
     */
    public static String convertToPinyinString(String text, String separator, boolean remainNone)
    {
        List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(text, remainNone);
        int length = pinyinList.size();
        StringBuilder sb = new StringBuilder(length * (5 + separator.length()));
        int i = 1;
        for (Pinyin pinyin : pinyinList)
        {
            sb.append(pinyin.getPinyinWithoutTone());
            if (i < length)
            {
                sb.append(separator);
            }
            ++i;
        }
        return sb.toString();
    }

    /**
     * 转化为拼音
     * @param text 代解析的文本
     * @return 一个拼音列表
     */
    public static List<Pinyin> convertToPinyinList(String text)
    {
        return PinyinDictionary.convertToPinyin(text);
    }

    /**
     * 转化为拼音（首字母）
     * @param text
     * @param separator
     * @param remainNone
     * @return
     */
    public static String convertToPinyinFirstCharString(String text, String separator, boolean remainNone)
    {
        List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(text, remainNone);
        int length = pinyinList.size();
        StringBuilder sb = new StringBuilder(length * (1 + separator.length()));
        int i = 1;
        for (Pinyin pinyin : pinyinList)
        {
            sb.append(pinyin.getFirstChar());
            if (i < length)
            {
                sb.append(separator);
            }
            ++i;
        }
        return sb.toString();
    }

    /**
     * 分词
     * @param text 文本
     * @return 切分后的单词
     */
    public static List<Term> segment(String text)
    {
        return StandTokenizer.segment(text);
    }

    /**
     * 创建一个分词器
     * @return
     */
    public static AbstractSegment createSegment()
    {
        return new Segment();
    }
}
