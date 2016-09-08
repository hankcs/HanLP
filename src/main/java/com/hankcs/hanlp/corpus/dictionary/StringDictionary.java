/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 19:53</create-date>
 *
 * <copyright file="StringDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary;


import com.hankcs.hanlp.corpus.io.IOUtil;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.AbstractMap;
import java.util.Map;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * 满足 key=value 格式的词典，其中“=”可以自定义
 * @author hankcs
 */
public class StringDictionary extends SimpleDictionary<String>
{
    /**
     * key value之间的分隔符
     */
    protected String separator;

    public StringDictionary(String separator)
    {
        this.separator = separator;
    }

    public StringDictionary()
    {
        this("=");
    }

    @Override
    protected Map.Entry<String, String> onGenerateEntry(String line)
    {
        String[] paramArray = line.split(separator, 2);
        if (paramArray.length != 2)
        {
            logger.warning("词典有一行读取错误： " + line);
            return null;
        }
        return new AbstractMap.SimpleEntry<String, String>(paramArray[0], paramArray[1]);
    }

    /**
     * 保存词典
     * @param path
     * @return 是否成功
     */
    public boolean save(String path)
    {
        try
        {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(IOUtil.newOutputStream(path)));
            for (Map.Entry<String, String> entry : trie.entrySet())
            {
                bw.write(entry.getKey());
                bw.write(separator);
                bw.write(entry.getValue());
                bw.newLine();
            }
            bw.close();
        }
        catch (Exception e)
        {
            logger.warning("保存词典到" + path + "失败");
            return true;
        }
        return false;
    }

    /**
     * 将自己逆转过来返回
     * @return
     */
    public StringDictionary reverse()
    {
        StringDictionary dictionary = new StringDictionary(separator);
        for (Map.Entry<String, String> entry : entrySet())
        {
            dictionary.trie.put(entry.getValue(), entry.getKey());
        }

        return dictionary;
    }
}
