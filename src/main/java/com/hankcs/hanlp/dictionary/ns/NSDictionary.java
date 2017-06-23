/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:39</create-date>
 *
 * <copyright file="NSDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.liNSunsoft.com/
 * This source is subject to the LiNSunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.ns;


import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.NS;
import com.hankcs.hanlp.dictionary.common.CommonDictionary;
import com.hankcs.hanlp.utility.ByteUtil;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 一个好用的地名词典
 *
 * @author hankcs
 */
public class NSDictionary extends CommonDictionary<EnumItem<NS>>
{
    @Override
    protected EnumItem<NS>[] onLoadValue(String path)
    {
        EnumItem<NS>[] valueArray = loadDat(path + ".value.dat");
        if (valueArray != null)
        {
            return valueArray;
        }
        List<EnumItem<NS>> valueList = new LinkedList<EnumItem<NS>>();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            String line;
            while ((line = br.readLine()) != null)
            {
                Map.Entry<String, Map.Entry<String, Integer>[]> args = EnumItem.create(line);
                EnumItem<NS> NSEnumItem = new EnumItem<NS>();
                for (Map.Entry<String, Integer> e : args.getValue())
                {
                    NSEnumItem.labelMap.put(NS.valueOf(e.getKey()), e.getValue());
                }
                valueList.add(NSEnumItem);
            }
            br.close();
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "失败" + e);
        }
        valueArray = valueList.toArray(new EnumItem[0]);
        return valueArray;
    }

    @Override
    protected boolean onSaveValue(EnumItem<NS>[] valueArray, String path)
    {
        return saveDat(path + ".value.dat", valueArray);
    }

    private EnumItem<NS>[] loadDat(String path)
    {
        byte[] bytes = IOUtil.readBytes(path);
        if (bytes == null) return null;
        NS[] NSArray = NS.values();
        int index = 0;
        int size = ByteUtil.bytesHighFirstToInt(bytes, index);
        index += 4;
        EnumItem<NS>[] valueArray = new EnumItem[size];
        for (int i = 0; i < size; ++i)
        {
            int currentSize = ByteUtil.bytesHighFirstToInt(bytes, index);
            index += 4;
            EnumItem<NS> item = new EnumItem<NS>();
            for (int j = 0; j < currentSize; ++j)
            {
                NS NS = NSArray[ByteUtil.bytesHighFirstToInt(bytes, index)];
                index += 4;
                int frequency = ByteUtil.bytesHighFirstToInt(bytes, index);
                index += 4;
                item.labelMap.put(NS, frequency);
            }
            valueArray[i] = item;
        }
        return valueArray;
    }

    private boolean saveDat(String path, EnumItem<NS>[] valueArray)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path));
            out.writeInt(valueArray.length);
            for (EnumItem<NS> item : valueArray)
            {
                out.writeInt(item.labelMap.size());
                for (Map.Entry<NS, Integer> entry : item.labelMap.entrySet())
                {
                    out.writeInt(entry.getKey().ordinal());
                    out.writeInt(entry.getValue());
                }
            }
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("保存失败" + e);
            return false;
        }
        return true;
    }
}
