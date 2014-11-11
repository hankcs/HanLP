/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/2 11:41</create-date>
 *
 * <copyright file="SYTDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.py;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.set.UnEmptyStringSet;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.io.IOUtil;

import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 声母韵母音调词典
 *
 * @author hankcs
 */
public class SYTDictionary
{
    static Set<String> smSet = new UnEmptyStringSet();
    static Set<String> ymSet = new UnEmptyStringSet();
    static Set<String> ydSet = new UnEmptyStringSet();
    static Map<String, String[]> map = new TreeMap<String, String[]>();

    static
    {
        StringDictionary dictionary = new StringDictionary();
        if (dictionary.load(HanLP.Config.SYTDictionaryPath))
        {
            logger.info("载入声母韵母音调词典" + HanLP.Config.SYTDictionaryPath + "成功");
            for (Map.Entry<String, String> entry : dictionary.entrySet())
            {
                //      0  1 2
                // bai1=b,ai,1
                String[] args = entry.getValue().split(",");
                if (args[0].length() == 0) args[0] = "none";
                smSet.add(args[0]);
                ymSet.add(args[1]);
                ydSet.add(args[2]);
                String[] valueArray = new String[4];
                System.arraycopy(args, 0, valueArray, 0, args.length);
                valueArray[3] = PinyinUtil.convertToneNumber2ToneMark(entry.getKey());
                map.put(entry.getKey(), valueArray);
            }
        }
        else
        {
            logger.warning("载入声母韵母音调词典" + HanLP.Config.SYTDictionaryPath + "失败");
        }
    }

    /**
     * 导出声母表等等
     *
     * @param path
     */
    public static void dumpEnum(String path)
    {
        dumpEnum(smSet, path + "sm.txt");
        dumpEnum(ymSet, path + "ym.txt");
        dumpEnum(ydSet, path + "yd.txt");
        Set<String> hdSet = new TreeSet<String>();
        for (Pinyin pinyin : PinyinDictionary.pinyins)
        {
            hdSet.add(pinyin.getHeadString());
        }
        dumpEnum(hdSet, path + "head.txt");
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String[]> entry : map.entrySet())
        {
            // 0声母 1韵母 2音调 3带音标
            String[] value = entry.getValue();
            Pinyin pinyin = Pinyin.valueOf(entry.getKey());
            sb.append(entry.getKey() + "(" + Shengmu.class.getSimpleName() + "." + value[0] + ", " + Yunmu.class.getSimpleName() + "." + value[1] + ", " + value[2] + ", \"" + value[3] + "\", \"" + entry.getKey().substring(0, entry.getKey().length() - 1)  + "\"" + ", " + Head.class.getSimpleName() + "." + pinyin.getHeadString() + ", '" + pinyin.getFirstChar() + "'" + "),\n");
        }
        IOUtil.saveTxt(path + "py.txt", sb.toString());
    }

    private static boolean dumpEnum(Set<String> set, String path)
    {
        StringBuilder sb = new StringBuilder();
        for (String s : set)
        {
            sb.append(s);
            sb.append(",\n");
        }
        return IOUtil.saveTxt(path, sb.toString());
    }
}
