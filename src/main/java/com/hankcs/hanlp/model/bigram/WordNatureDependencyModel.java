/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 15:00</create-date>
 *
 * <copyright file="WordNatureDependencyModel.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.bigram;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.dependency.model.WordNatureWeightModelMaker;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dependency.common.Edge;
import com.hankcs.hanlp.dependency.common.Node;
import com.hankcs.hanlp.utility.Predefine;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 词、词性相互构成依存关系的统计句法分析模型
 * @author hankcs
 */
public class WordNatureDependencyModel
{
    DoubleArrayTrie<Attribute> trie;

    public WordNatureDependencyModel(String path)
    {
        long start = System.currentTimeMillis();
        if (load(path))
        {
            logger.info("加载依存句法生成模型" + path + "成功，耗时：" + (System.currentTimeMillis() - start) + " ms");
        }
        else
        {
            throw new IllegalArgumentException("加载依存句法生成模型" + path + "失败，耗时：" + (System.currentTimeMillis() - start) + " ms");
        }
    }

    boolean load(String path)
    {
        trie = new DoubleArrayTrie<Attribute>();
        if (loadDat(path)) return true;
        TreeMap<String, Attribute> map = new TreeMap<String, Attribute>();
        TreeMap<String, Integer> tagMap = new TreeMap<String, Integer>();
        for (String line : IOUtil.readLineListWithLessMemory(path))
        {
            String[] param = line.split(" ");
            if (param[0].endsWith("@"))
            {
                tagMap.put(param[0], Integer.parseInt(param[2]));
                continue;
            }
            int natureCount = (param.length - 1) / 2;
            Attribute attribute = new Attribute(natureCount);
            for (int i = 0; i < natureCount; ++i)
            {
                attribute.dependencyRelation[i] = param[1 + 2 * i];
                attribute.p[i] = Integer.parseInt(param[2 + 2 * i]);
            }
            map.put(param[0], attribute);
        }
        if (map.size() == 0) return false;
        // 为它们计算概率
        for (Map.Entry<String, Attribute> entry : map.entrySet())
        {
            String key = entry.getKey();
            String[] param = key.split("@", 2);
            Attribute attribute = entry.getValue();
            int total = tagMap.get(param[0] + "@");
            for (int i = 0; i < attribute.p.length; ++i)
            {
                attribute.p[i] = (float) -Math.log(attribute.p[i] / total);
            }
            // 必须降低平滑处理的权重
            float boost = 1.0f;
            if (key.startsWith("<"))
            {
                boost *= 10.0f;
            }
            if (key.endsWith(">"))
            {
                boost *= 10.0f;
            }
            if (boost != 1.0f)
                attribute.setBoost(boost);
        }

        trie.build(map);
        if (!saveDat(path, map)) logger.warning("缓存" + path + "失败");
        return true;
    }

    boolean saveDat(String path, TreeMap<String, Attribute> map)
    {
        Collection<Attribute> attributeList = map.values();
        // 缓存值文件
        try
        {
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path + Predefine.BIN_EXT));
            out.writeInt(attributeList.size());
            for (Attribute attribute : attributeList)
            {
                out.writeInt(attribute.p.length);
                for (int i = 0; i < attribute.p.length; ++i)
                {
                    char[] charArray = attribute.dependencyRelation[i].toCharArray();
                    out.writeInt(charArray.length);
                    for (char c : charArray)
                    {
                        out.writeChar(c);
                    }
                    out.writeFloat(attribute.p[i]);
                }
            }
            if (!trie.save(out)) return false;
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("保存失败" + e);
            return false;
        }
        return true;
    }

    boolean loadDat(String path)
    {
        ByteArray byteArray = ByteArray.createByteArray(path + Predefine.BIN_EXT);
        if (byteArray == null) return false;
        int size = byteArray.nextInt();
        Attribute[] attributeArray = new Attribute[size];
        for (int i = 0; i < attributeArray.length; ++i)
        {
            int length = byteArray.nextInt();
            Attribute attribute = new Attribute(length);
            for (int j = 0; j < attribute.dependencyRelation.length; ++j)
            {
                attribute.dependencyRelation[j] = byteArray.nextString();
                attribute.p[j] = byteArray.nextFloat();
            }
            attributeArray[i] = attribute;
        }

        return trie.load(byteArray, attributeArray);
    }

    public Attribute get(String key)
    {
        return trie.get(key);
    }

    /**
     * 打分
     * @param from
     * @param to
     * @return
     */
    public Edge getEdge(Node from, Node to)
    {
        // 首先尝试词+词
        Attribute attribute = get(from.compiledWord, to.compiledWord);
        if (attribute == null) attribute = get(from.compiledWord, WordNatureWeightModelMaker.wrapTag(to.label));
        if (attribute == null) attribute = get(WordNatureWeightModelMaker.wrapTag(from.label), to.compiledWord);
        if (attribute == null) attribute = get(WordNatureWeightModelMaker.wrapTag(from.label), WordNatureWeightModelMaker.wrapTag(to.label));
        if (attribute == null)
        {
            attribute = Attribute.NULL;
        }
        if (HanLP.Config.DEBUG)
        {
            System.out.println(from + " 到 " + to + " : " + attribute);
        }
        return new Edge(from.id, to.id, attribute.dependencyRelation[0], attribute.p[0]);
    }

    public Attribute get(String from, String to)
    {
        return get(from + "@" + to);
    }

    static class Attribute
    {
        final static Attribute NULL = new Attribute("未知", 10000.0f);
        /**
         * 依存关系
         */
        public String[] dependencyRelation;
        /**
         * 概率
         */
        public float[] p;

        public Attribute(int size)
        {
            dependencyRelation = new String[size];
            p = new float[size];
        }

        Attribute(String dr, float p)
        {
            dependencyRelation = new String[]{dr};
            this.p = new float[]{p};
        }

        /**
         * 加权
         * @param boost
         */
        public void setBoost(float boost)
        {
            for (int i = 0; i < p.length; ++i)
            {
                p[i] *= boost;
            }
        }

        @Override
        public String toString()
        {
            final StringBuilder sb = new StringBuilder(dependencyRelation.length * 10);
            for (int i = 0; i < dependencyRelation.length; ++i)
            {
                sb.append(dependencyRelation[i]).append(' ').append(p[i]).append(' ');
            }
            return sb.toString();
        }
    }
}
