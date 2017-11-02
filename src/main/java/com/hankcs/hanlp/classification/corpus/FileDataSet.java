/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/20 PM4:42</create-date>
 *
 * <copyright file="FileDataSet.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.corpus;

import com.hankcs.hanlp.classification.collections.FrequencyMap;
import com.hankcs.hanlp.classification.models.AbstractModel;

import java.io.*;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * @author hankcs
 */
public class FileDataSet extends AbstractDataSet
{
    File cache;
    DataOutputStream out;
    int size;

    public FileDataSet(AbstractModel model, File cache) throws FileNotFoundException
    {
        super(model);
        initCache(cache);
    }

    public FileDataSet(AbstractModel model) throws IOException
    {
        this(model, File.createTempFile(String.valueOf(System.currentTimeMillis()), ".dat"));
    }

    public FileDataSet(File cache) throws FileNotFoundException
    {
        initCache(cache);
    }

    private void initCache(File cache) throws FileNotFoundException
    {
        this.cache = cache;
        out = new DataOutputStream(new FileOutputStream(cache));
    }

    private void initCache() throws IOException
    {
        initCache(File.createTempFile(String.valueOf(System.currentTimeMillis()), ".dat"));
    }

    public FileDataSet() throws IOException
    {
        this(File.createTempFile(String.valueOf(System.currentTimeMillis()), ".dat"));
    }

    @Override
    public Document add(String category, String text)
    {
        Document document = convert(category, text);
        try
        {
            add(document);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
        return document;
    }

    private void add(Document document) throws IOException
    {
        out.writeInt(document.category);
        Set<Map.Entry<Integer, int[]>> entrySet = document.tfMap.entrySet();
        out.writeInt(entrySet.size());
        for (Map.Entry<Integer, int[]> entry : entrySet)
        {
            out.writeInt(entry.getKey());
            out.writeInt(entry.getValue()[0]);
        }
        ++size;
    }

    @Override
    public int size()
    {
        return size;
    }

    @Override
    public void clear()
    {
        size = 0;
    }

    @Override
    public IDataSet shrink(int[] idMap)
    {
        try
        {
            clear();
            Iterator<Document> iterator = iterator();
            initCache();
            while (iterator.hasNext())
            {
                Document document = iterator.next();
                FrequencyMap<Integer> tfMap = new FrequencyMap<Integer>();
                for (Map.Entry<Integer, int[]> entry : document.tfMap.entrySet())
                {
                    Integer feature = entry.getKey();
                    if (idMap[feature] == -1) continue;
                    tfMap.put(idMap[feature], entry.getValue());
                }
                // 检查是否是空白文档
                if (tfMap.size() == 0) continue;
                document.tfMap = tfMap;
                add(document);
            }
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }

        return this;
    }

    @Override
    public Iterator<Document> iterator()
    {
        try
        {
            out.close();
            final DataInputStream in  = new DataInputStream(new FileInputStream(cache));
            return new Iterator<Document>()
            {
                @Override
                public void remove()
                {
                    throw new RuntimeException("不支持的操作");
                }

                @Override
                public boolean hasNext()
                {
                    try
                    {
                        boolean next = in.available() > 0;
                        if (!next) in.close();
                        return next;
                    }
                    catch (IOException e)
                    {
                        throw new RuntimeException(e);
                    }
                }

                @Override
                public Document next()
                {
                    try
                    {
                        return new Document(in);
                    }
                    catch (IOException e)
                    {
                        throw new RuntimeException(e);
                    }
                }
            };
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }
}
