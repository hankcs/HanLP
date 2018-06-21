/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM5:23</create-date>
 *
 * <copyright file="FeatureMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;
import com.hankcs.hanlp.model.perceptron.common.IStringIdMap;
import com.hankcs.hanlp.model.perceptron.common.TaskType;
import com.hankcs.hanlp.model.perceptron.tagset.CWSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.POSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.Set;

/**
 * @author hankcs
 */
public abstract class FeatureMap implements IStringIdMap, ICacheAble
{
    public abstract int size();

    public int[] allLabels()
    {
        return tagSet.allTags();
    }

    public int bosTag()
    {
        return tagSet.size();
    }

    public TagSet tagSet;
    /**
     * 是否允许新增特征
     */
    public boolean mutable;

    public FeatureMap(TagSet tagSet)
    {
        this(tagSet, false);
    }

    public FeatureMap(TagSet tagSet, boolean mutable)
    {
        this.tagSet = tagSet;
        this.mutable = mutable;
    }

    public abstract Set<Map.Entry<String, Integer>> entrySet();

    public FeatureMap(boolean mutable)
    {
        this.mutable = mutable;
    }

    public FeatureMap()
    {
        this(false);
    }

    @Override
    public void save(DataOutputStream out) throws IOException
    {
        tagSet.save(out);
        out.writeInt(size());
        for (Map.Entry<String, Integer> entry : entrySet())
        {
            out.writeUTF(entry.getKey());
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        loadTagSet(byteArray);
        int size = byteArray.nextInt();
        for (int i = 0; i < size; i++)
        {
            idOf(byteArray.nextUTF());
        }
        return true;
    }

    protected final void loadTagSet(ByteArray byteArray)
    {
        TaskType type = TaskType.values()[byteArray.nextInt()];
        switch (type)
        {
            case CWS:
                tagSet = new CWSTagSet();
                break;
            case POS:
                tagSet = new POSTagSet();
                break;
            case NER:
                tagSet = new NERTagSet();
                break;
            case CLASSIFICATION:
                tagSet = new TagSet(TaskType.CLASSIFICATION);
                break;
        }
        tagSet.load(byteArray);
    }
}