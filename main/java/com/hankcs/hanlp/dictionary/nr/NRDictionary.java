/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:39</create-date>
 *
 * <copyright file="NRDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.nr;


import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.dictionary.CommonDictionary;
import com.hankcs.hanlp.utility.Utility;

import java.io.*;
import java.util.AbstractMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 一个好用的人名词典
 *
 * @author hankcs
 */
public class NRDictionary extends CommonDictionary<EnumItem<NR>>
{
    @Override
    protected EnumItem<NR>[] onLoadValue(String path)
    {
        EnumItem<NR>[] valueArray = loadDat(path + ".value.dat");
        if (valueArray != null)
        {
            return valueArray;
        }
        List<EnumItem<NR>> valueList = new LinkedList<>();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                Map.Entry<String, Map.Entry<String, Integer>[]> args = EnumItem.create(line);
                EnumItem<NR> nrEnumItem = new EnumItem<>();
                for (Map.Entry<String, Integer> e : args.getValue())
                {
                    nrEnumItem.labelMap.put(NR.valueOf(e.getKey()), e.getValue());
                }
                valueList.add(nrEnumItem);
            }
            br.close();
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "失败" + e);
        }
        valueArray = valueList.toArray(new EnumItem[0]);
        saveDat(path + ".value.dat", valueArray);
        return valueArray;
    }

    private EnumItem<NR>[] loadDat(String path)
    {
        byte[] bytes = IOUtil.readBytes(path);
        if (bytes == null) return null;
        NR[] nrArray = NR.values();
        int index = 0;
        int size = Utility.bytesHighFirstToInt(bytes, index);
        index += 4;
        EnumItem<NR>[] valueArray = new EnumItem[size];
        for (int i = 0; i < size; ++i)
        {
            int currentSize = Utility.bytesHighFirstToInt(bytes, index);
            index += 4;
            EnumItem<NR> item = new EnumItem<>();
            for (int j = 0; j < currentSize; ++j)
            {
                NR nr = nrArray[Utility.bytesHighFirstToInt(bytes, index)];
                index += 4;
                int frequency = Utility.bytesHighFirstToInt(bytes, index);
                index += 4;
                item.labelMap.put(nr, frequency);
            }
            valueArray[i] = item;
        }
        return valueArray;
    }

    private boolean saveDat(String path, EnumItem<NR>[] valueArray)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(new FileOutputStream(path));
            out.writeInt(valueArray.length);
            for (EnumItem<NR> item : valueArray)
            {
                out.writeInt(item.labelMap.size());
                for (Map.Entry<NR, Integer> entry : item.labelMap.entrySet())
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
