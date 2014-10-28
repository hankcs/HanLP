/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/15 19:38</create-date>
 *
 * <copyright file="StopwordDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.AbstractMap;
import java.util.Map;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class StopWordDictionary extends CommonDictionary<Boolean>
{
    @Override
    protected Boolean[] onLoadValue(String path)
    {
        int size = 0;
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                ++size;
            }
            br.close();
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "失败" + e);
        }
        Boolean[] valueArray = new Boolean[size];
        for (int i = 0; i < valueArray.length; ++i)
        {
            valueArray[i] = true;
        }

        return valueArray;
    }
}
