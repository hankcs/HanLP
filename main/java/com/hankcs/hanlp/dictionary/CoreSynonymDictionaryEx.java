/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 13:12</create-date>
 *
 * <copyright file="CoreSynonymDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algoritm.EditDistance;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.NShort.Path.WordResult;
import com.hankcs.hanlp.utility.Utility;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * 核心同义词词典
 *
 * @author hankcs
 */
public class CoreSynonymDictionaryEx
{
    static CommonSynonymDictionaryEx dictionary;

    static
    {
        try
        {
            dictionary = CommonSynonymDictionaryEx.create(new FileInputStream(HanLP.Config.CoreSynonymDictionaryDictionaryPath));
        }
        catch (Exception e)
        {
            logger.warning("载入核心同义词词典失败" + e);
        }
    }

    public static Long[] get(String key)
    {
        return dictionary.get(key);
    }

    /**
     * 语义距离
     * @param itemA
     * @param itemB
     * @return
     */
    public static long distance(CommonSynonymDictionary.SynonymItem itemA, CommonSynonymDictionary.SynonymItem itemB)
    {
        return itemA.distance(itemB);
    }

    /**
     * 将分词结果转换为同义词列表
     * @param sentence 句子
     * @param withUndefinedItem 是否保留词典中没有的词语
     * @return
     */
    public static List<Long[]> convert(List<WordResult> sentence, boolean withUndefinedItem)
    {
        List<Long[]> synonymItemList = new ArrayList<>(sentence.size());
        for (WordResult wordResult : sentence)
        {
            // 除掉停用词
            if (wordResult.nPOS == null) continue;
            String nature = wordResult.nPOS.toString();
            char firstChar = nature.charAt(0);
            switch (firstChar)
            {
                case 'm':
                {
                    if (!Utility.isAllChinese(wordResult.sWord)) continue;
                }break;
                case 'w':
                {
                    continue;
                }
            }
            // 停用词
            if (CoreStopWordDictionary.contains(wordResult.sWord)) continue;
            Long[] item = get(wordResult.sWord);
//            logger.trace("{} {}", wordResult.sWord, Arrays.toString(item));
            if (item == null)
            {
                if (withUndefinedItem)
                {
                    item = new Long[]{Long.MAX_VALUE / 3};
                    synonymItemList.add(item);
                }

            }
            else
            {
                synonymItemList.add(item);
            }
        }

        return synonymItemList;
    }

    /**
     * 获取语义标签
     * @return
     */
    public static long[] getLexemeArray(List<CommonSynonymDictionary.SynonymItem> synonymItemList)
    {
        long[] array = new long[synonymItemList.size()];
        int i = 0;
        for (CommonSynonymDictionary.SynonymItem item : synonymItemList)
        {
            array[i++] = item.entry.id;
        }
        return array;
    }


    public long distance(List<CommonSynonymDictionary.SynonymItem> synonymItemListA, List<CommonSynonymDictionary.SynonymItem> synonymItemListB)
    {
        return EditDistance.compute(synonymItemListA, synonymItemListB);
    }

    public long distance(long[] arrayA, long[] arrayB)
    {
        return EditDistance.compute(arrayA, arrayB);
    }
}
