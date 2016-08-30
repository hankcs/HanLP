/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/4/23 23:00</create-date>
 *
 * <copyright file="TestCharTable.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2015, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.other.CharTable;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Map;

/**
 * @author hankcs
 */
public class TestCharTable extends TestCase
{
    public void testConvert() throws Exception
    {
        System.out.println(CharTable.CONVERT['關']);
        System.out.println(CharTable.CONVERT['Ａ']);
        System.out.println(CharTable.CONVERT['“']);
        System.out.println(CharTable.CONVERT['．']);
    }

    public void testEnd() throws Exception
    {
        System.out.println(CharTable.CONVERT['，']);
        System.out.println(CharTable.CONVERT['。']);
        System.out.println(CharTable.CONVERT['！']);
        System.out.println(CharTable.CONVERT['…']);
    }

    public void testFix() throws Exception
    {
        char[] CONVERT = CharTable.CONVERT;
        CONVERT['.'] = '.';
        CONVERT['．'] = '.';
        CONVERT['。'] = '.';
        CONVERT['！'] = '!';
        CONVERT['，'] = ',';
        CONVERT['!'] = '!';
        CONVERT['#'] = '#';
        CONVERT['&'] = '&';
        CONVERT['*'] = '*';
        CONVERT[','] = ',';
        CONVERT['/'] = '/';
        CONVERT[';'] = ';';
        CONVERT['?'] = '?';
        CONVERT['\\'] = '\\';
        CONVERT['^'] = '^';
        CONVERT['_'] = '_';
        CONVERT['`'] = '`';
        CONVERT['|'] = '|';
        CONVERT['~'] = '~';
        CONVERT['¡'] = '¡';
        CONVERT['¦'] = '¦';
        CONVERT['´'] = '´';
        CONVERT['¸'] = '¸';
        CONVERT['¿'] = '¿';
        CONVERT['ˇ'] = 'ˇ';
        CONVERT['ˉ'] = 'ˉ';
        CONVERT['ˊ'] = 'ˊ';
        CONVERT['ˋ'] = 'ˋ';
        CONVERT['˜'] = '˜';
        CONVERT['—'] = '—';
        CONVERT['―'] = '―';
        CONVERT['‖'] = '‖';
        CONVERT['…'] = '…';
        CONVERT['∕'] = '∕';
        CONVERT['︳'] = '︳';
        CONVERT['︴'] = '︴';
        CONVERT['﹉'] = '﹉';
        CONVERT['﹊'] = '﹊';
        CONVERT['﹋'] = '﹋';
        CONVERT['﹌'] = '﹌';
        CONVERT['﹍'] = '﹍';
        CONVERT['﹎'] = '﹎';
        CONVERT['﹏'] = '﹏';
        CONVERT['﹐'] = '﹐';
        CONVERT['﹑'] = '﹑';
        CONVERT['﹔'] = '﹔';
        CONVERT['﹖'] = '﹖';
        CONVERT['﹟'] = '﹟';
        CONVERT['﹠'] = '﹠';
        CONVERT['﹡'] = '﹡';
        CONVERT['﹨'] = '﹨';
        CONVERT['＇'] = '＇';
        CONVERT['；'] = '；';
        CONVERT['？'] = '？';
        CONVERT['幣'] = '币';
        CONVERT['繫'] = '系';
        CONVERT['眾'] = '众';
        CONVERT['龕'] = '龛';
        CONVERT['製'] = '制';
        for (int i = 0; i < CONVERT.length; i++)
        {
            if (CONVERT[i] == '\u0000')
            {
                if (i != '\u0000') CONVERT[i] = (char) i;
                else CONVERT[i] = ' ';
            }
        }
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(HanLP.Config.CharTablePath));
        out.writeObject(CONVERT);
        out.close();
    }

    public void testImportSingleCharFromTraditionalChineseDictionary() throws Exception
    {
//        char[] CONVERT = CharTable.CONVERT;
//        StringDictionary dictionary = new StringDictionary("=");
//        dictionary.load(HanLP.Config.t2sDictionaryPath);
//        for (Map.Entry<String, String> entry : dictionary.entrySet())
//        {
//            String key = entry.getKey();
//            if (key.length() != 1) continue;
//            String value = entry.getValue();
//            char t = key.charAt(0);
//            char s = value.charAt(0);
////            if (CONVERT[t] != s)
////            {
////                System.out.printf("%s\t%c=%c\n", entry, t, CONVERT[t]);
////            }
//            CONVERT[t] = s;
//        }
//
//        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(HanLP.Config.CharTablePath));
//        out.writeObject(CONVERT);
//        out.close();
    }

    public void testDumpCharTable() throws Exception
    {
        BufferedWriter bw = IOUtil.newBufferedWriter(HanLP.Config.CharTablePath.replace(".bin.yes", ".txt"));
        char[] CONVERT = CharTable.CONVERT;
        for (int i = 0; i < CONVERT.length; i++)
        {
            if (i != CONVERT[i])
            {
                bw.write(String.format("%c=%c\n", i, CONVERT[i]));
            }
        }
        bw.close();
    }

    public void testLoadCharTableFromTxt() throws Exception
    {
//        CharTable.load(HanLP.Config.CharTablePath.replace(".bin.yes", ".txt"));
    }
}
