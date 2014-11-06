/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/6 16:18</create-date>
 *
 * <copyright file="TestXianDaiHanYu.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.SimpleDictionary;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.dictionary.StringDictionaryMaker;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.py.Pinyin;
import com.hankcs.hanlp.dictionary.py.PinyinDictionary;
import com.hankcs.hanlp.dictionary.py.TonePinyinString2PinyinConverter;
import com.hankcs.hanlp.dictionary.ts.TraditionalChineseDictionary;
import junit.framework.TestCase;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author hankcs
 */
public class TestXianDaiHanYu extends TestCase
{
    public void testMakeDictionary() throws Exception
    {
        String text = IOUtil.readTxt("D:\\Doc\\语料库\\现代汉语词典（第五版）全文_更新.txt").toLowerCase();
        Pattern pattern = Pattern.compile("【([\\u4E00-\\u9FA5]+)】([abcdefghijklmnopqrstuwxyzāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ∥•’－]+)");
        Matcher matcher = pattern.matcher(text);
        StringDictionary dictionary = new StringDictionary();
        while (matcher.find())
        {
            String word = matcher.group(1);
            String pinyinString = matcher.group(2);
            List<Pinyin> pinyinList = TonePinyinString2PinyinConverter.convert(pinyinString, false);
            if (pinyinList.size() != word.length() || hasNull(pinyinList))
            {
                System.out.println("转换失败 " + word + " " + pinyinString + " " + pinyinList);
                continue;
            }
            dictionary.add(word, convertList2String(pinyinList));
        }
        System.out.println(dictionary.size());
        dictionary.save("data/dictionary/pinyin/pinyin.xd.txt");
    }

    public void testMakePyDictionary() throws Exception
    {
        StringDictionary dictionaryRaw = new StringDictionary();
        dictionaryRaw.load("D:\\PythonProjects\\python-pinyin\\dic.txt");

        StringDictionary dictionary = new StringDictionary();
        for (Map.Entry<String, String> entry : dictionaryRaw.entrySet())
        {
            String word = entry.getKey();
            String[] pinyinArray = entry.getValue().split(",");
            List<Pinyin> pinyinList = TonePinyinString2PinyinConverter.convert(pinyinArray);
            if (word.length() != pinyinList.size() || hasNull(pinyinList))
            {
                System.out.println(entry + " | " + pinyinList);
            }
            else
            {
                dictionary.add(entry.getKey(), convertList2String(pinyinList));
            }
        }

        dictionary.save("data/dictionary/pinyin/pinyin.python.txt");
    }

    public void testCombinePy() throws Exception
    {
        StringDictionary dictionary = new StringDictionary();
        dictionary.load("data/dictionary/pinyin/pinyin.python.txt");
        dictionary.remove(new SimpleDictionary.Filter<String>()
        {
            @Override
            public boolean remove(Map.Entry<String, String> entry)
            {
                String key = entry.getKey();
                String[] pinyinArray = entry.getValue().split(",");
                List<Pinyin> pinyinList = TonePinyinString2PinyinConverter.convertFromToneNumber(pinyinArray);
                // 检查是否实用
                List<Pinyin> localPinyinList = PinyinDictionary.convertToPinyin(key);
                if (!isEqual(pinyinList, localPinyinList))
                {
                    System.out.println("接受 " + key + "=" + pinyinList + "!=" + localPinyinList);
                    return false;
                }
                else
                {
                    return true;
                }
            }
        });

        StringDictionary dictionaryLocal = new StringDictionary();
        dictionaryLocal.load(HanLP.Config.PinyinDictionaryPath);
        dictionaryLocal.combine(dictionary);
        dictionaryLocal.save(HanLP.Config.PinyinDictionaryPath);
    }

    public void testMakeKaiFangDictionary() throws Exception
    {
        // data/dictionary/tc/
        LinkedList<String> lineList = IOUtil.readLineList("D:\\Doc\\语料库\\cidian_zhzh-kfcd-2013122.txt");
        StringDictionary dictionaryKFTC = new StringDictionary();
        for (String line : lineList)
        {
            String[] args = line.split("\\s");
            // 愛面子	爱面子	ai4 mian4 zi5
            List<Pinyin> pinyinList = new ArrayList<>(args.length - 2);
            for (int i = 2; i < args.length; ++i)
            {
                pinyinList.add(TonePinyinString2PinyinConverter.convertFromToneNumber(args[i]));
            }
            if (hasNull(pinyinList) || pinyinList.size() != args[1].length())
            {
//                System.out.println("忽略 " + line + " " + pinyinList);
                continue;
            }
            // 检查是否实用
            List<Pinyin> localPinyinList = PinyinDictionary.convertToPinyin(args[1]);
            if (!isEqual(pinyinList, localPinyinList))
            {
                System.out.println("接受 " + args[1] + "=" + pinyinList + "!=" + localPinyinList);
                dictionaryKFTC.add(args[1], convertList2String(pinyinList));
            }
        }

        StringDictionary dictionaryLocal = new StringDictionary();
        dictionaryLocal.load(HanLP.Config.PinyinDictionaryPath);
        dictionaryLocal.combine(dictionaryKFTC);
        dictionaryLocal.save(HanLP.Config.PinyinDictionaryPath);
    }

    public void testPinyin() throws Exception
    {
        System.out.println(PinyinDictionary.convertToPinyin("龟背"));

    }

    private boolean isEqual(List<Pinyin> pinyinListA, List<Pinyin> pinyinListB)
    {
        if (pinyinListA.size() != pinyinListB.size()) return false;

        Iterator<Pinyin> iteratorA = pinyinListA.iterator();
        Iterator<Pinyin> iteratorB = pinyinListB.iterator();
        while (iteratorA.hasNext())
        {
            if (iteratorA.next() != iteratorB.next()) return false;
        }

        return true;
    }

    public void testT2C() throws Exception
    {
        System.out.println(TraditionalChineseDictionary.convertToSimplifiedChinese("熱線"));

    }

    public void testConvertSingle() throws Exception
    {
        System.out.println(TonePinyinString2PinyinConverter.convert("ai"));
    }

    private String convertList2String(List<Pinyin> pinyinList)
    {
        StringBuilder sb = new StringBuilder();
        for (Pinyin pinyin : pinyinList)
        {
            sb.append(pinyin);
            sb.append(',');
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    private boolean hasNull(List<Pinyin> pinyinList)
    {
        for (Pinyin pinyin : pinyinList)
        {
            if (pinyin == null) return true;
        }

        return false;
    }

    public void testEnumChar() throws Exception
    {
        Set<Character> characterSet = new TreeSet<>();
        for (Pinyin pinyin : PinyinDictionary.pinyins)
        {
            for (char c : pinyin.getPinyinWithToneMark().toCharArray())
            {
                characterSet.add(c);
            }
        }

        for (Character c : characterSet)
        {
            System.out.print(c);
        }
    }

    public void testToken() throws Exception
    {
        System.out.println(TonePinyinString2PinyinConverter.convert("āgōng", true));
    }
}
