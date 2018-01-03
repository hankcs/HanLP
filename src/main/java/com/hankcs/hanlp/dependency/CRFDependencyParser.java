/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/11 21:09</create-date>
 *
 * <copyright file="CRFDependencyParser.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dependency.common.POSUtil;
import com.hankcs.hanlp.model.bigram.BigramDependencyModel;
import com.hankcs.hanlp.model.crf.CRFModel;
import com.hankcs.hanlp.model.crf.FeatureFunction;
import com.hankcs.hanlp.model.crf.Table;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;
import com.hankcs.hanlp.utility.GlobalObjectPool;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 基于随机条件场的依存句法分析器
 *
 * @deprecated 关于将线性CRF序列标注应用于句法分析，我持反对意见。CRF的链式结构决定它的视野只有当前位置的前后n个单词构成的特征，
 * 如果依存节点恰好落在这n个范围内还好理解，如果超出该范围，利用这个n个单词的特征推测它是不合理的。
 * 也就是说，我认为利用链式CRF预测长依存是不科学的。线性链CRF做句法分析的理论基础非常薄弱，一阶CRF这个标注模型根本无法阻止环的产生，
 * 这份实现也没有复现论文的结果，所以不再维护，其模型文件也不再打包到新data里面。请使用在理论和工程上更稳定的
 * {@link com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser}。
 *
 * @author hankcs
 */
public class CRFDependencyParser extends AbstractDependencyParser
{
    CRFModel crfModel;

    public CRFDependencyParser(String modelPath)
    {
        crfModel = GlobalObjectPool.get(modelPath);
        if (crfModel != null) return;
        long start = System.currentTimeMillis();
        if (load(modelPath))
        {
            logger.info("加载随机条件场依存句法分析器模型" + modelPath + "成功，耗时 " + (System.currentTimeMillis() - start) + " ms");
            GlobalObjectPool.put(modelPath, crfModel);
        }
        else
        {
            logger.info("加载随机条件场依存句法分析器模型" + modelPath + "失败，耗时 " + (System.currentTimeMillis() - start) + " ms");
        }
    }

    public CRFDependencyParser()
    {
        this(HanLP.Config.CRFDependencyModelPath);
    }

    /**
     * 分析句子的依存句法
     *
     * @param termList 句子，可以是任何具有词性标注功能的分词器的分词结果
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(List<Term> termList)
    {
        return new CRFDependencyParser().parse(termList);
    }

    /**
     * 分析句子的依存句法
     *
     * @param sentence 句子
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(String sentence)
    {
        return new CRFDependencyParser().parse(sentence);
    }

    boolean load(String path)
    {
        if (loadDat(path + Predefine.BIN_EXT)) return true;
        crfModel = CRFModel.loadTxt(path, new CRFModelForDependency(new DoubleArrayTrie<FeatureFunction>())); // 使用特化版的CRF
        return crfModel != null;
    }

    boolean loadDat(String path)
    {
        ByteArray byteArray = ByteArray.createByteArray(path);
        if (byteArray == null) return false;
        crfModel = new CRFModelForDependency(new DoubleArrayTrie<FeatureFunction>());
        return crfModel.load(byteArray);
    }

    boolean saveDat(String path)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path));
            crfModel.save(out);
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("在缓存" + path + "时发生错误" + TextUtility.exceptionToString(e));
            return false;
        }

        return true;
    }

    @Override
    public CoNLLSentence parse(List<Term> termList)
    {
        Table table = new Table();
        table.v = new String[termList.size()][4];
        Iterator<Term> iterator = termList.iterator();
        for (String[] line : table.v)
        {
            Term term = iterator.next();
            line[0] = term.word;
            line[2] = POSUtil.compilePOS(term.nature);
            line[1] = line[2].substring(0, 1);
        }
        crfModel.tag(table);
        if (HanLP.Config.DEBUG)
        {
            System.out.println(table);
        }
        CoNLLWord[] coNLLWordArray = new CoNLLWord[table.size()];
        for (int i = 0; i < coNLLWordArray.length; i++)
        {
            coNLLWordArray[i] = new CoNLLWord(i + 1, table.v[i][0], table.v[i][2], table.v[i][1]);
        }
        int i = 0;
        for (String[] line : table.v)
        {
            CRFModelForDependency.DTag dTag = new CRFModelForDependency.DTag(line[3]);
            if (dTag.pos.endsWith("ROOT"))
            {
                coNLLWordArray[i].HEAD = CoNLLWord.ROOT;
            }
            else
            {
                int index = convertOffset2Index(dTag, table, i);
                if (index == -1)
                    coNLLWordArray[i].HEAD = CoNLLWord.NULL;
                else coNLLWordArray[i].HEAD = coNLLWordArray[index];
            }
            ++i;
        }

        for (i = 0; i < coNLLWordArray.length; i++)
        {
            coNLLWordArray[i].DEPREL = BigramDependencyModel.get(coNLLWordArray[i].NAME, coNLLWordArray[i].POSTAG, coNLLWordArray[i].HEAD.NAME, coNLLWordArray[i].HEAD.POSTAG);
        }
        return new CoNLLSentence(coNLLWordArray);
    }

    static int convertOffset2Index(CRFModelForDependency.DTag dTag, Table table, int current)
    {
        int posCount = 0;
        if (dTag.offset > 0)
        {
            for (int i = current + 1; i < table.size(); ++i)
            {
                if (table.v[i][1].equals(dTag.pos)) ++posCount;
                if (posCount == dTag.offset) return i;
            }
        }
        else
        {
            for (int i = current - 1; i >= 0; --i)
            {
                if (table.v[i][1].equals(dTag.pos)) ++posCount;
                if (posCount == -dTag.offset) return i;
            }
        }

        return -1;
    }

    /**
     * 必须对维特比算法做一些特化修改
     */
    static class CRFModelForDependency extends CRFModel
    {

        public CRFModelForDependency(ITrie<FeatureFunction> featureFunctionTrie)
        {
            super(featureFunctionTrie);
        }

        /**
         * 每个tag的分解。内部类的内部类你到底累不累
         */
        static class DTag
        {
            int offset;
            String pos;

            public DTag(String tag)
            {
                String[] args = tag.split("_", 2);
                if (args[0].charAt(0) == '+') args[0] = args[0].substring(1);
                offset = Integer.parseInt(args[0]);
                pos = args[1];
            }

            @Override
            public String toString()
            {
                return (offset > 0 ? "+" : "") + offset + "_" + pos;
            }
        }

        DTag[] id2dtag;

        @Override
        public boolean load(ByteArray byteArray)
        {
            if (!super.load(byteArray)) return false;
            initId2dtagArray();
            return true;
        }

        private void initId2dtagArray()
        {
            id2dtag = new DTag[id2tag.length];
            for (int i = 0; i < id2tag.length; i++)
            {
                id2dtag[i] = new DTag(id2tag[i]);
            }
        }

        @Override
        protected void onLoadTxtFinished()
        {
            super.onLoadTxtFinished();
            initId2dtagArray();
        }

        boolean isLegal(int tagId, int current, Table table)
        {
            DTag tag = id2dtag[tagId];
            if ("ROOT".equals(tag.pos))
            {
                for (int i = 0; i < current; ++i)
                {
                    if (table.v[i][3].endsWith("ROOT")) return false;
                }
                return true;
            }
            else
            {
                int posCount = 0;
                if (tag.offset > 0)
                {
                    for (int i = current + 1; i < table.size(); ++i)
                    {
                        if (table.v[i][1].equals(tag.pos)) ++posCount;
                        if (posCount == tag.offset) return true;
                    }
                    return false;
                }
                else
                {
                    for (int i = current - 1; i >= 0; --i)
                    {
                        if (table.v[i][1].equals(tag.pos)) ++posCount;
                        if (posCount == -tag.offset) return true;
                    }
                    return false;
                }
            }
        }

        @Override
        public void tag(Table table)
        {
            int size = table.size();
            double bestScore = Double.MIN_VALUE;
            int bestTag = 0;
            int tagSize = id2tag.length;
            LinkedList<double[]> scoreList = computeScoreList(table, 0);    // 0位置命中的特征函数
            for (int i = 0; i < tagSize; ++i)   // -1位置的标签遍历
            {
                for (int j = 0; j < tagSize; ++j)   // 0位置的标签遍历
                {
                    if (!isLegal(j, 0, table)) continue;
                    double curScore = computeScore(scoreList, j);
                    if (matrix != null)
                    {
                        curScore += matrix[i][j];
                    }
                    if (curScore > bestScore)
                    {
                        bestScore = curScore;
                        bestTag = j;
                    }
                }
            }
            table.setLast(0, id2tag[bestTag]);
            int preTag = bestTag;
            // 0位置打分完毕，接下来打剩下的
            for (int i = 1; i < size; ++i)
            {
                scoreList = computeScoreList(table, i);    // i位置命中的特征函数
                bestScore = Double.MIN_VALUE;
                for (int j = 0; j < tagSize; ++j)   // i位置的标签遍历
                {
                    if (!isLegal(j, i, table)) continue;
                    double curScore = computeScore(scoreList, j);
                    if (matrix != null)
                    {
                        curScore += matrix[preTag][j];
                    }
                    if (curScore > bestScore)
                    {
                        bestScore = curScore;
                        bestTag = j;
                    }
                }
                table.setLast(i, id2tag[bestTag]);
                preTag = bestTag;
            }
        }
    }
}
