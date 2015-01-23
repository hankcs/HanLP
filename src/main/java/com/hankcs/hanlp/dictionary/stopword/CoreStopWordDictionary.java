/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/15 19:39</create-date>
 *
 * <copyright file="CoreStopwordDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.stopword;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.Predefine;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.logging.Level;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 核心停用词词典
 * @author hankcs
 */
public class CoreStopWordDictionary
{
    static StopWordDictionary dictionary;
    static
    {
        dictionary = new StopWordDictionary();
        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.CoreStopWordDictionaryPath + Predefine.BIN_EXT);
        if (byteArray == null)
        {
            try
            {
                dictionary = new StopWordDictionary(new File(HanLP.Config.CoreStopWordDictionaryPath));
                DataOutputStream out = new DataOutputStream(new FileOutputStream(HanLP.Config.CoreStopWordDictionaryPath + Predefine.BIN_EXT));
                dictionary.save(out);
                out.close();
            }
            catch (Exception e)
            {
                logger.log(Level.SEVERE, "载入停用词词典" + HanLP.Config.CoreStopWordDictionaryPath + "失败", e);
            }
        }
        else
        {
            dictionary.load(byteArray);
        }
    }

    public static boolean contains(String key)
    {
        return dictionary.contains(key);
    }

    /**
     * 核心停用词典的核心过滤器
     */
    public static Filter FILTER = new Filter()
    {
        @Override
        public boolean shouldInclude(Term term)
        {
            return CoreStopWordDictionary.shouldInclude(term);
        }
    };

    /**
     * 是否应当将这个term纳入计算，词性属于名词、动词、副词、形容词，并且不在停用词表中
     *
     * @param term
     * @return 是否应当
     */
    public static boolean shouldInclude(Term term)
    {
        // 除掉停用词
        if (term.nature == null) return false;
        String nature = term.nature.toString();
        char firstChar = nature.charAt(0);
        switch (firstChar)
        {
            case 'm':
            case 'b':
            case 'c':
            case 'e':
            case 'o':
            case 'p':
            case 'q':
            case 'u':
            case 'y':
            case 'z':
            case 'r':
            case 'w':
            {
                return false;
            }
            default:
            {
                if (term.word.length() > 1 && !CoreStopWordDictionary.contains(term.word))
                {
                    return true;
                }
            }
            break;
        }

        return false;
    }

}
