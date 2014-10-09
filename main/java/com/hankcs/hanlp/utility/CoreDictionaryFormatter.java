/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/07/2014/7/2 17:08</create-date>
 *
 * <copyright file="CoreDictionaryFormatter.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.dictionary.CoreDictionary;

import java.io.*;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * 重新定制核心词典的格式
 * @author hankcs
 */
public class CoreDictionaryFormatter
{
    static void format()
    {
        BufferedReader br = null;
        BufferedWriter bw = null;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream("data/dictionary/CoreDictionary.txt")));
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/dictionary/CoreDictionary_new.txt")));
            String line;
            while ((line = br.readLine()) != null)
            {
                String param[] = line.split("\t");
                String word = param[0];
                CoreDictionary.Attribute attribute = null;
                String natureStr = param[1].substring(1, param[1].length() - 1);
                String[] split = natureStr.split(",");
                String[] str;
                Map<Integer, String> frequencyNatureMap = new TreeMap<Integer, String>(Collections.reverseOrder());
                for (int i = 0; i < split.length; ++i)
                {
                    str = split[i].split("=");
                    frequencyNatureMap.put(Integer.parseInt(str[1]), str[0].trim());
                }

                bw.write(word);
                for (Map.Entry<Integer, String> entry : frequencyNatureMap.entrySet())
                {
                    bw.write(" " + entry.getValue() + " " + entry.getKey());
                }
                bw.newLine();
            }
            br.close();
            bw.close();
        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        } catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public static void main(String[] args)
    {
        format();
    }
}
