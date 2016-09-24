/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/25 12:42</create-date>
 *
 * <copyright file="Model.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.maxent;

import com.hankcs.hanlp.collection.dartsclone.Pair;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 最大熵模型，采用双数组Trie树加速，值得拥有
 *
 * @author hankcs
 */
public class MaxEntModel
{
    /**
     * 常数C，训练的时候用到
     */
    int correctionConstant;
    /**
     * 为修正特征函数对应的参数，在预测的时候并没有用到
     */
    double correctionParam;
    /**
     * 归一化
     */
    UniformPrior prior;
    /**
     * 事件名
     */
    protected String[] outcomeNames;
    /**
     * 衡量参数
     */
    EvalParameters evalParams;

    /**
     * 将特征与一个数字（下标）对应起来的映射map
     */
    DoubleArrayTrie<Integer> pmap;

    /**
     * 预测分布
     *
     * @param context 环境
     * @return 概率数组
     */
    public final double[] eval(String[] context)
    {
        return (eval(context, new double[evalParams.getNumOutcomes()]));
    }

    /**
     * 预测分布
     *
     * @param context
     * @return
     */
    public final List<Pair<String, Double>> predict(String[] context)
    {
        List<Pair<String, Double>> result = new ArrayList<Pair<String, Double>>(outcomeNames.length);
        double[] p = eval(context);
        for (int i = 0; i < p.length; ++i)
        {
            result.add(new Pair<String, Double>(outcomeNames[i], p[i]));
        }
        return result;
    }

    /**
     * 预测概率最高的分类
     *
     * @param context
     * @return
     */
    public final Pair<String, Double> predictBest(String[] context)
    {
        List<Pair<String, Double>> resultList = predict(context);
        double bestP = -1.0;
        Pair<String, Double> bestPair = null;
        for (Pair<String, Double> pair : resultList)
        {
            if (pair.getSecond() > bestP)
            {
                bestP = pair.getSecond();
                bestPair = pair;
            }
        }

        return bestPair;
    }

    /**
     * 预测分布
     *
     * @param context
     * @return
     */
    public final List<Pair<String, Double>> predict(Collection<String> context)
    {
        return predict(context.toArray(new String[0]));
    }

    /**
     * 预测分布
     *
     * @param context 环境
     * @param outsums 先验分布
     * @return 概率数组
     */
    public final double[] eval(String[] context, double[] outsums)
    {
        assert context != null;
        int[] scontexts = new int[context.length];
        for (int i = 0; i < context.length; i++)
        {
            Integer ci = pmap.get(context[i]);
            scontexts[i] = ci == null ? -1 : ci;
        }
        prior.logPrior(outsums);
        return eval(scontexts, outsums, evalParams);
    }

    /**
     * 预测
     * @param context 环境
     * @param prior 先验概率
     * @param model 特征函数
     * @return
     */
    public static double[] eval(int[] context, double[] prior, EvalParameters model)
    {
        Context[] params = model.getParams();
        int numfeats[] = new int[model.getNumOutcomes()];
        int[] activeOutcomes;
        double[] activeParameters;
        double value = 1;
        for (int ci = 0; ci < context.length; ci++)
        {
            if (context[ci] >= 0)
            {
                Context predParams = params[context[ci]];
                activeOutcomes = predParams.getOutcomes();
                activeParameters = predParams.getParameters();
                for (int ai = 0; ai < activeOutcomes.length; ai++)
                {
                    int oid = activeOutcomes[ai];
                    numfeats[oid]++;
                    prior[oid] += activeParameters[ai] * value;
                }
            }
        }

        double normal = 0.0;
        for (int oid = 0; oid < model.getNumOutcomes(); oid++)
        {
            if (model.getCorrectionParam() != 0)
            {
                prior[oid] = Math
                        .exp(prior[oid]
                                     * model.getConstantInverse()
                                     + ((1.0 - ((double) numfeats[oid] / model
                                .getCorrectionConstant())) * model.getCorrectionParam()));
            }
            else
            {
                prior[oid] = Math.exp(prior[oid] * model.getConstantInverse());
            }
            normal += prior[oid];
        }

        for (int oid = 0; oid < model.getNumOutcomes(); oid++)
        {
            prior[oid] /= normal;
        }
        return prior;
    }

    /**
     * 从文件加载，同时缓存为二进制文件
     * @param path
     * @return
     */
    public static MaxEntModel create(String path)
    {
        MaxEntModel m = new MaxEntModel();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path + Predefine.BIN_EXT));
            br.readLine();  // type
            m.correctionConstant = Integer.parseInt(br.readLine());  // correctionConstant
            out.writeInt(m.correctionConstant);
            m.correctionParam = Double.parseDouble(br.readLine());  // getCorrectionParameter
            out.writeDouble(m.correctionParam);
            // label
            int numOutcomes = Integer.parseInt(br.readLine());
            out.writeInt(numOutcomes);
            String[] outcomeLabels = new String[numOutcomes];
            m.outcomeNames = outcomeLabels;
            for (int i = 0; i < numOutcomes; i++)
            {
                outcomeLabels[i] = br.readLine();
                TextUtility.writeString(outcomeLabels[i], out);
            }
            // pattern
            int numOCTypes = Integer.parseInt(br.readLine());
            out.writeInt(numOCTypes);
            int[][] outcomePatterns = new int[numOCTypes][];
            for (int i = 0; i < numOCTypes; i++)
            {
                StringTokenizer tok = new StringTokenizer(br.readLine(), " ");
                int[] infoInts = new int[tok.countTokens()];
                out.writeInt(infoInts.length);
                for (int j = 0; tok.hasMoreTokens(); j++)
                {
                    infoInts[j] = Integer.parseInt(tok.nextToken());
                    out.writeInt(infoInts[j]);
                }
                outcomePatterns[i] = infoInts;
            }
            // feature
            int NUM_PREDS = Integer.parseInt(br.readLine());
            out.writeInt(NUM_PREDS);
            String[] predLabels = new String[NUM_PREDS];
            m.pmap = new DoubleArrayTrie<Integer>();
            TreeMap<String, Integer> tmpMap = new TreeMap<String, Integer>();
            for (int i = 0; i < NUM_PREDS; i++)
            {
                predLabels[i] = br.readLine();
                assert !tmpMap.containsKey(predLabels[i]) : "重复的键： " + predLabels[i] + " 请使用 -Dfile.encoding=UTF-8 训练";
                TextUtility.writeString(predLabels[i], out);
                tmpMap.put(predLabels[i], i);
            }
            m.pmap.build(tmpMap);
            for (Map.Entry<String, Integer> entry : tmpMap.entrySet())
            {
                out.writeInt(entry.getValue());
            }
            m.pmap.save(out);
            // params
            Context[] params = new Context[NUM_PREDS];
            int pid = 0;
            for (int i = 0; i < outcomePatterns.length; i++)
            {
                int[] outcomePattern = new int[outcomePatterns[i].length - 1];
                for (int k = 1; k < outcomePatterns[i].length; k++)
                {
                    outcomePattern[k - 1] = outcomePatterns[i][k];
                }
                for (int j = 0; j < outcomePatterns[i][0]; j++)
                {
                    double[] contextParameters = new double[outcomePatterns[i].length - 1];
                    for (int k = 1; k < outcomePatterns[i].length; k++)
                    {
                        contextParameters[k - 1] = Double.parseDouble(br.readLine());
                        out.writeDouble(contextParameters[k - 1]);
                    }
                    params[pid] = new Context(outcomePattern, contextParameters);
                    pid++;
                }
            }
            // prior
            m.prior = new UniformPrior();
            m.prior.setLabels(outcomeLabels);
            // eval
            m.evalParams = new EvalParameters(params, m.correctionParam, m.correctionConstant, outcomeLabels.length);
            out.close();
        }
        catch (Exception e)
        {
            logger.severe("从" + path + "加载最大熵模型失败！" + TextUtility.exceptionToString(e));
            return null;
        }
        return m;
    }

    /**
     * 从字节流快速加载
     * @param byteArray
     * @return
     */
    public static MaxEntModel create(ByteArray byteArray)
    {
        MaxEntModel m = new MaxEntModel();
        m.correctionConstant = byteArray.nextInt();  // correctionConstant
        m.correctionParam = byteArray.nextDouble();  // getCorrectionParameter
        // label
        int numOutcomes = byteArray.nextInt();
        String[] outcomeLabels = new String[numOutcomes];
        m.outcomeNames = outcomeLabels;
        for (int i = 0; i < numOutcomes; i++) outcomeLabels[i] = byteArray.nextString();
        // pattern
        int numOCTypes = byteArray.nextInt();
        int[][] outcomePatterns = new int[numOCTypes][];
        for (int i = 0; i < numOCTypes; i++)
        {
            int length = byteArray.nextInt();
            int[] infoInts = new int[length];
            for (int j = 0; j < length; j++)
            {
                infoInts[j] = byteArray.nextInt();
            }
            outcomePatterns[i] = infoInts;
        }
        // feature
        int NUM_PREDS = byteArray.nextInt();
        String[] predLabels = new String[NUM_PREDS];
        m.pmap = new DoubleArrayTrie<Integer>();
        for (int i = 0; i < NUM_PREDS; i++)
        {
            predLabels[i] = byteArray.nextString();
        }
        Integer[] v = new Integer[NUM_PREDS];
        for (int i = 0; i < v.length; i++)
        {
            v[i] = byteArray.nextInt();
        }
        m.pmap.load(byteArray, v);
        // params
        Context[] params = new Context[NUM_PREDS];
        int pid = 0;
        for (int i = 0; i < outcomePatterns.length; i++)
        {
            int[] outcomePattern = new int[outcomePatterns[i].length - 1];
            for (int k = 1; k < outcomePatterns[i].length; k++)
            {
                outcomePattern[k - 1] = outcomePatterns[i][k];
            }
            for (int j = 0; j < outcomePatterns[i][0]; j++)
            {
                double[] contextParameters = new double[outcomePatterns[i].length - 1];
                for (int k = 1; k < outcomePatterns[i].length; k++)
                {
                    contextParameters[k - 1] = byteArray.nextDouble();
                }
                params[pid] = new Context(outcomePattern, contextParameters);
                pid++;
            }
        }
        // prior
        m.prior = new UniformPrior();
        m.prior.setLabels(outcomeLabels);
        // eval
        m.evalParams = new EvalParameters(params, m.correctionParam, m.correctionConstant, outcomeLabels.length);
        return m;
    }

    /**
     * 加载最大熵模型<br>
     *     如果存在缓存的话，优先读取缓存，否则读取txt，并且建立缓存
     * @param txtPath txt的路径，即使不存在.txt，只存在.bin，也应传入txt的路径，方法内部会自动加.bin后缀
     * @return
     */
    public static MaxEntModel load(String txtPath)
    {
        ByteArray byteArray = ByteArray.createByteArray(txtPath + Predefine.BIN_EXT);
        if (byteArray != null) return create(byteArray);
        return create(txtPath);
    }
}
