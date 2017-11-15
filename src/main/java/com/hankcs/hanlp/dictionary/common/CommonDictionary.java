/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 22:30</create-date>
 *
 * <copyright file="CommonDictioanry.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.common;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.BIN_EXT;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 通用的词典，对应固定格式的词典，但是标签可以泛型化
 *
 * @author hankcs
 */
public abstract class CommonDictionary<V>
{
    DoubleArrayTrie<V> trie;

    protected abstract V[] loadValueArray(ByteArray byteArray);

    public boolean load(String path)
    {
        trie = new DoubleArrayTrie<V>();
        long start = System.currentTimeMillis();
        if (loadDat(ByteArray.createByteArray(path + BIN_EXT)))
        {
            return true;
        }
        TreeMap<String, V> map = new TreeMap<String, V>();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] paramArray = line.split("\\s");
                map.put(paramArray[0], createValue(paramArray));
            }
            br.close();
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "失败" + e);
        }
        Set<Map.Entry<String, V>> entrySet = map.entrySet();
        List<String> keyList = new ArrayList<String>(entrySet.size());
        List<V> valueList = new ArrayList<V>(entrySet.size());
        for (Map.Entry<String, V> entry : entrySet)
        {
            keyList.add(entry.getKey());
            valueList.add(entry.getValue());
        }
        int resultCode = trie.build(keyList, valueList);
        if (resultCode != 0)
        {
            logger.warning("trie建立失败");
            return false;
        }
        logger.info(path + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        saveDat(path + BIN_EXT, valueList);
        return true;
    }

    protected boolean loadDat(ByteArray byteArray)
    {
        V[] valueArray = loadValueArray(byteArray);
        if (valueArray == null)
        {
            return false;
        }
        return trie.load(byteArray, valueArray);
    }

    protected boolean saveDat(String path, List<V> valueArray)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path));
            out.writeInt(valueArray.size());
            for (V item : valueArray)
            {
                saveValue(item, out);
            }
            trie.save(out);
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("保存失败" + TextUtility.exceptionToString(e));
            return false;
        }
        return true;
    }

    protected abstract void saveValue(V value, DataOutputStream out) throws IOException;

    /**
     * 查询一个单词
     *
     * @param key
     * @return 单词对应的条目
     */
    public V get(String key)
    {
        return trie.get(key);
    }

    /**
     * 是否含有键
     *
     * @param key
     * @return
     */
    public boolean contains(String key)
    {
        return get(key) != null;
    }

    /**
     * 词典大小
     *
     * @return
     */
    public int size()
    {
        return trie.size();
    }

    protected abstract V createValue(String[] params);
}
