/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 22:19</create-date>
 *
 * <copyright file="PlaySuggester.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.suggest.ISuggester;
import com.hankcs.hanlp.suggest.Suggester;
import junit.framework.TestCase;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;

/**
 * @author hankcs
 */
public class PlaySuggester extends TestCase
{
    public static void main(String[] args) throws Exception
    {
        testSuggest();
    }
    public static void testSuggest() throws Exception
    {
        ISuggester suggester = new Suggester();
//        load("data/title.txt", suggester);
        load("data/phrase.txt", suggester);
        String[] testCaseArray = new String[]
                {
//                        "护照丢了",
                        "中国人民",
                        "zhongguorenmin",
                        "zgrm",
                        "zgrenmin",
                        "中国renmin",
                        "租房",
                        "假日安排",
                        "身份证丢了",
                        "就医",
                        "孩子上学",
                        "教室资格", // 就算用户输了错别字，也可以矫正一部分
                        "教育",
                        "生育",
                };
        for (String key : testCaseArray)
        {
            runCase(suggester, key);
        }
        Scanner scanner = new Scanner(System.in);
        String line;
        while ((line = scanner.nextLine()).length() > 0
                )
        {
            runCase(suggester, line);
        }
    }

    public static void load(String path, ISuggester iSuggester) throws IOException
    {
        String line;
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            while ((line = br.readLine()) != null)
            {
                line = line.trim();
                line = line.split("\\s")[0];
                if (line.length() <= 3 || line.length() > 20) continue;
//                System.out.println("正在读入并处理 " + line);
                iSuggester.addSentence(line);
            }
            br.close();
        }
    }

    public static void runCase(ISuggester ISuggester, String key)
    {
        long start = System.currentTimeMillis();
        System.out.println(key + " " + ISuggester.suggest(key, 10) + " " + (System.currentTimeMillis() - start) + "ms");
    }
}
