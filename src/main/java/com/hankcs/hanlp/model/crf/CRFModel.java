/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 20:53</create-date>
 *
 * <copyright file="Model.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.BIN_EXT;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class CRFModel implements ICacheAble
{
    /**
     * 标签和id的相互转换
     */
    Map<String, Integer> tag2id;
    /**
     * id转标签
     */
    protected String[] id2tag;
    /**
     * 特征函数
     */
    ITrie<FeatureFunction> featureFunctionTrie;
    /**
     * 特征模板
     */
    List<FeatureTemplate> featureTemplateList;
    /**
     * tag的二元转移矩阵，适用于BiGram Feature
     */
    protected double[][] matrix;

    public CRFModel()
    {
        featureFunctionTrie = new DoubleArrayTrie<FeatureFunction>();
    }

    /**
     * 以指定的trie树结构储存内部特征函数
     * @param featureFunctionTrie
     */
    public CRFModel(ITrie<FeatureFunction> featureFunctionTrie)
    {
        this.featureFunctionTrie = featureFunctionTrie;
    }

    protected void onLoadTxtFinished()
    {
        // do nothing
    }

    /**
     * 加载Txt形式的CRF++模型
     * @param path 模型路径
     * @param instance 模型的实例（这里允许用户构造不同的CRFModel来储存最终读取的结果）
     * @return 该模型
     */
    public static CRFModel loadTxt(String path, CRFModel instance)
    {
        CRFModel CRFModel = instance;
        // 先尝试从bin加载
        if (CRFModel.load(ByteArray.createByteArray(path + Predefine.BIN_EXT))) return CRFModel;
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(path);
        if (!lineIterator.hasNext()) return null;
        logger.info(lineIterator.next());   // verson
        logger.info(lineIterator.next());   // cost-factor
        int maxid = Integer.parseInt(lineIterator.next().substring("maxid:".length()).trim());
        logger.info(lineIterator.next());   // xsize
        lineIterator.next();    // blank
        String line;
        int id = 0;
        CRFModel.tag2id = new HashMap<String, Integer>();
        while ((line = lineIterator.next()).length() != 0)
        {
            CRFModel.tag2id.put(line, id);
            ++id;
        }
        CRFModel.id2tag = new String[CRFModel.tag2id.size()];
        final int size = CRFModel.id2tag.length;
        for (Map.Entry<String, Integer> entry : CRFModel.tag2id.entrySet())
        {
            CRFModel.id2tag[entry.getValue()] = entry.getKey();
        }
        TreeMap<String, FeatureFunction> featureFunctionMap = new TreeMap<String, FeatureFunction>();  // 构建trie树的时候用
        List<FeatureFunction> featureFunctionList = new LinkedList<FeatureFunction>(); // 读取权值的时候用
        CRFModel.featureTemplateList = new LinkedList<FeatureTemplate>();
        while ((line = lineIterator.next()).length() != 0)
        {
            if (!"B".equals(line))
            {
                FeatureTemplate featureTemplate = FeatureTemplate.create(line);
                CRFModel.featureTemplateList.add(featureTemplate);
            }
            else
            {
                CRFModel.matrix = new double[size][size];
            }
        }

        if (CRFModel.matrix != null)
        {
            lineIterator.next();    // 0 B
        }

        while ((line = lineIterator.next()).length() != 0)
        {
            String[] args = line.split(" ", 2);
            char[] charArray = args[1].toCharArray();
            FeatureFunction featureFunction = new FeatureFunction(charArray, size);
            featureFunctionMap.put(args[1], featureFunction);
            featureFunctionList.add(featureFunction);
        }

        if (CRFModel.matrix != null)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    CRFModel.matrix[i][j] = Double.parseDouble(lineIterator.next());
                }
            }
        }

        for (FeatureFunction featureFunction : featureFunctionList)
        {
            for (int i = 0; i < size; i++)
            {
                featureFunction.w[i] = Double.parseDouble(lineIterator.next());
            }
        }
        if (lineIterator.hasNext())
        {
            logger.warning("文本读取有残留，可能会出问题！" + path);
        }
        lineIterator.close();
        logger.info("开始构建trie树");
        CRFModel.featureFunctionTrie.build(featureFunctionMap);
        // 缓存bin
        try
        {
            logger.info("开始缓存" + path + Predefine.BIN_EXT);
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path + Predefine.BIN_EXT));
            CRFModel.save(out);
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("在缓存" + path + Predefine.BIN_EXT + "时发生错误" + TextUtility.exceptionToString(e));
        }
        CRFModel.onLoadTxtFinished();
        return CRFModel;
    }

    /**
     * 维特比后向算法标注
     *
     * @param table
     */
    public void tag(Table table)
    {
        int size = table.size();
        if (size == 0) return;
        int tagSize = id2tag.length;
        double[][] net = new double[size][tagSize];
        for (int i = 0; i < size; ++i)
        {
            LinkedList<double[]> scoreList = computeScoreList(table, i);
            for (int tag = 0; tag < tagSize; ++tag)
            {
                net[i][tag] = computeScore(scoreList, tag);
            }
        }

        if (size == 1)
        {
            double maxScore = -1e10;
            int bestTag = 0;
            for (int tag = 0; tag < net[0].length; ++tag)
            {
                if (net[0][tag] > maxScore)
                {
                    maxScore = net[0][tag];
                    bestTag = tag;
                }
            }
            table.setLast(0, id2tag[bestTag]);
            return;
        }

        int[][] from = new int[size][tagSize];
        double[][] maxScoreAt = new double[2][tagSize]; // 滚动数组
        System.arraycopy(net[0], 0, maxScoreAt[0], 0, tagSize); // 初始preI=0,  maxScoreAt[preI][pre] = net[0][pre]
        int curI = 0;
        for (int i = 1; i < size; ++i)
        {
            curI = i & 1;
            int preI = 1 - curI;
            for (int now = 0; now < tagSize; ++now)
            {
                double maxScore = -1e10;
                for (int pre = 0; pre < tagSize; ++pre)
                {
                    double score = maxScoreAt[preI][pre] + matrix[pre][now] + net[i][now];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        from[i][now] = pre;
                        maxScoreAt[curI][now] = maxScore;
                    }
                }
                net[i][now] = maxScore;
            }
        }
        // 反向回溯最佳路径
        double maxScore = -1e10;
        int maxTag = 0;
        for (int tag = 0; tag < tagSize; ++tag)
        {
            if (maxScoreAt[curI][tag] > maxScore)
            {
                maxScore = maxScoreAt[curI][tag];
                maxTag = tag;
            }
        }

        table.setLast(size - 1, id2tag[maxTag]);
        maxTag = from[size - 1][maxTag];
        for (int i = size - 2; i > 0; --i)
        {
            table.setLast(i, id2tag[maxTag]);
            maxTag = from[i][maxTag];
        }
        table.setLast(0, id2tag[maxTag]);
    }

    /**
     * 根据特征函数计算输出
     * @param table
     * @param current
     * @return
     */
    protected LinkedList<double[]> computeScoreList(Table table, int current)
    {
        LinkedList<double[]> scoreList = new LinkedList<double[]>();
        for (FeatureTemplate featureTemplate : featureTemplateList)
        {
            char[] o = featureTemplate.generateParameter(table, current);
            FeatureFunction featureFunction = featureFunctionTrie.get(o);
            if (featureFunction == null) continue;
            scoreList.add(featureFunction.w);
        }

        return scoreList;
    }

    /**
     * 给一系列特征函数结合tag打分
     *
     * @param scoreList
     * @param tag
     * @return
     */
    protected static double computeScore(LinkedList<double[]> scoreList, int tag)
    {
        double score = 0;
        for (double[] w : scoreList)
        {
            score += w[tag];
        }
        return score;
    }

    @Override
    public void save(DataOutputStream out) throws Exception
    {
        out.writeInt(id2tag.length);
        for (String tag : id2tag)
        {
            out.writeUTF(tag);
        }
        FeatureFunction[] valueArray = featureFunctionTrie.getValueArray(new FeatureFunction[0]);
        out.writeInt(valueArray.length);
        for (FeatureFunction featureFunction : valueArray)
        {
            featureFunction.save(out);
        }
        featureFunctionTrie.save(out);
        out.writeInt(featureTemplateList.size());
        for (FeatureTemplate featureTemplate : featureTemplateList)
        {
            featureTemplate.save(out);
        }
        if (matrix != null)
        {
            out.writeInt(matrix.length);
            for (double[] line : matrix)
            {
                for (double v : line)
                {
                    out.writeDouble(v);
                }
            }
        }
        else
        {
            out.writeInt(0);
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        if (byteArray == null) return false;
        try
        {
            int size = byteArray.nextInt();
            id2tag = new String[size];
            tag2id = new HashMap<String, Integer>(size);
            for (int i = 0; i < id2tag.length; i++)
            {
                id2tag[i] = byteArray.nextUTF();
                tag2id.put(id2tag[i], i);
            }
            FeatureFunction[] valueArray = new FeatureFunction[byteArray.nextInt()];
            for (int i = 0; i < valueArray.length; i++)
            {
                valueArray[i] = new FeatureFunction();
                valueArray[i].load(byteArray);
            }
            featureFunctionTrie.load(byteArray, valueArray);
            size = byteArray.nextInt();
            featureTemplateList = new ArrayList<FeatureTemplate>(size);
            for (int i = 0; i < size; ++i)
            {
                FeatureTemplate featureTemplate = new FeatureTemplate();
                featureTemplate.load(byteArray);
                featureTemplateList.add(featureTemplate);
            }
            size = byteArray.nextInt();
            if (size == 0) return true;
            matrix = new double[size][size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i][j] = byteArray.nextDouble();
                }
            }
        }
        catch (Exception e)
        {
            logger.warning("缓存载入失败，可能是由于版本变迁带来的不兼容。具体异常是：\n" + TextUtility.exceptionToString(e));
            return false;
        }

        return true;
    }

    /**
     * 加载Txt形式的CRF++模型<br>
     *     同时生成path.bin模型缓存
     * @param path 模型路径
     * @return 该模型
     */
    public static CRFModel loadTxt(String path)
    {
        return loadTxt(path, new CRFModel(new DoubleArrayTrie<FeatureFunction>()));
    }

    /**
     * 加载CRF++模型<br>
     *     如果存在缓存的话，优先读取缓存，否则读取txt，并且建立缓存
     * @param path txt的路径，即使不存在.txt，只存在.bin，也应传入txt的路径，方法内部会自动加.bin后缀
     * @return
     */
    public static CRFModel load(String path)
    {
        CRFModel model = loadBin(path + BIN_EXT);
        if (model != null) return model;
        return loadTxt(path, new CRFModel(new DoubleArrayTrie<FeatureFunction>()));
    }

    /**
     * 加载Bin形式的CRF++模型<br>
     *     注意该Bin形式不是CRF++的二进制模型,而是HanLP由CRF++的文本模型转换过来的私有格式
     * @param path
     * @return
     */
    public static CRFModel loadBin(String path)
    {
        ByteArray byteArray = ByteArray.createByteArray(path);
        if (byteArray == null) return null;
        CRFModel model = new CRFModel();
        if (model.load(byteArray)) return model;
        return null;
    }

    /**
     * 获取某个tag的ID
     * @param tag
     * @return
     */
    public Integer getTagId(String tag)
    {
        return tag2id.get(tag);
    }
}
