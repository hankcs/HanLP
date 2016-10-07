/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 17:24</create-date>
 *
 * <copyright file="WordNatureDependencyParser.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.dependency.common.Edge;
import com.hankcs.hanlp.dependency.common.Node;
import com.hankcs.hanlp.model.bigram.WordNatureDependencyModel;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;
import com.hankcs.hanlp.utility.GlobalObjectPool;

import java.util.List;

/**
 * 一个简单的句法分析器
 *
 * @author hankcs
 */
public class WordNatureDependencyParser extends MinimumSpanningTreeParser
{
    private WordNatureDependencyModel model;

    public WordNatureDependencyParser(WordNatureDependencyModel model)
    {
        this.model = model;
    }

    public WordNatureDependencyParser(String modelPath)
    {
        model = GlobalObjectPool.get(modelPath);
        if (model != null) return;
        model = new WordNatureDependencyModel(modelPath);
        GlobalObjectPool.put(modelPath, model);
    }

    public WordNatureDependencyParser()
    {
        this(HanLP.Config.WordNatureModelPath);
    }

    /**
     * 分析句子的依存句法
     *
     * @param termList 句子，可以是任何具有词性标注功能的分词器的分词结果
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(List<Term> termList)
    {
        return new WordNatureDependencyParser().parse(termList);
    }

    /**
     * 分析句子的依存句法
     *
     * @param sentence 句子
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(String sentence)
    {
        return new WordNatureDependencyParser().parse(sentence);
    }

    @Override
    protected Edge makeEdge(Node[] nodeArray, int from, int to)
    {
        return model.getEdge(nodeArray[from], nodeArray[to]);
    }
}
