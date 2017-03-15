/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/11 16:26</create-date>
 *
 * <copyright file="FamilyName.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.nr;

import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;

import java.io.*;
import java.util.List;

/**
 * @author hankcs
 */
public class FamilyName
{
    static boolean fn[];
    static
    {
        fn = new boolean[65535];
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("data/dictionary/person/familyname.txt")));
            String line;
            while ((line = br.readLine()) != null)
            {
                fn[line.charAt(0)] = true;
            }
            br.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public static boolean contains(char c)
    {
        return fn[c];
    }

    public static boolean contains(String c)
    {
        if (c.length() != 1) return false;
        return fn[c.charAt(0)];
    }
}
