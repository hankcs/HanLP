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
import com.hankcs.hanlp.collection.dartsclone.Pair;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ByteArrayFileStream;
import com.hankcs.hanlp.dependency.common.Edge;
import com.hankcs.hanlp.dependency.common.Node;
import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;
import com.hankcs.hanlp.model.maxent.MaxEntModel;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.GlobalObjectPool;
import com.hankcs.hanlp.utility.Predefine;

import java.util.LinkedList;
import java.util.List;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 最大熵句法分析器
 *
 * @deprecated 已废弃，请使用{@link KBeamArcEagerDependencyParser}。未来版本将不再发布该模型，并删除配置项
 * @author hankcs
 */
public class MaxEntDependencyParser extends MinimumSpanningTreeParser
{
    private MaxEntModel model;

    public MaxEntDependencyParser(MaxEntModel model)
    {
        this.model = model;
    }

    public MaxEntDependencyParser()
    {
        String path = HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT;
        model = GlobalObjectPool.get(path);
        if (model != null) return;
        long start = System.currentTimeMillis();
        ByteArray byteArray = ByteArrayFileStream.createByteArrayFileStream(path);
        if (byteArray != null)
        {
            model = MaxEntModel.create(byteArray);
        }
        else
        {
            model = MaxEntModel.create(HanLP.Config.MaxEntModelPath);
        }
        if (model != null)
        {
            GlobalObjectPool.put(path, model);
        }
        String result = model == null ? "失败" : "成功";
        logger.info("最大熵依存句法模型载入" + result + "，耗时" + (System.currentTimeMillis() - start) + " ms");
    }

    /**
     * 分析句子的依存句法
     *
     * @param termList 句子，可以是任何具有词性标注功能的分词器的分词结果
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(List<Term> termList)
    {
        return new MaxEntDependencyParser().parse(termList);
    }

    /**
     * 分析句子的依存句法
     *
     * @param sentence 句子
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(String sentence)
    {
        return new MaxEntDependencyParser().parse(sentence);
    }

    @Override
    protected Edge makeEdge(Node[] nodeArray, int from, int to)
    {
        LinkedList<String> context = new LinkedList<String>();
        int index = from;
        for (int i = index - 2; i < index + 2 + 1; ++i)
        {
            Node w = i >= 0 && i < nodeArray.length ? nodeArray[i] : Node.NULL;
            context.add(w.compiledWord + "i" + (i - index));      // 在尾巴上做个标记，不然特征冲突了
            context.add(w.label + "i" + (i - index));
        }
        index = to;
        for (int i = index - 2; i < index + 2 + 1; ++i)
        {
            Node w = i >= 0 && i < nodeArray.length ? nodeArray[i] : Node.NULL;
            context.add(w.compiledWord + "j" + (i - index));      // 在尾巴上做个标记，不然特征冲突了
            context.add(w.label + "j" + (i - index));
        }
        context.add(nodeArray[from].compiledWord + '→' + nodeArray[to].compiledWord);
        context.add(nodeArray[from].label + '→' + nodeArray[to].label);
        context.add(nodeArray[from].compiledWord + '→' + nodeArray[to].compiledWord + (from - to));
        context.add(nodeArray[from].label + '→' + nodeArray[to].label + (from - to));
        Node wordBeforeI = from - 1 >= 0 ? nodeArray[from - 1] : Node.NULL;
        Node wordBeforeJ = to - 1 >= 0 ? nodeArray[to - 1] : Node.NULL;
        context.add(wordBeforeI.compiledWord + '@' + nodeArray[from].compiledWord + '→' + nodeArray[to].compiledWord);
        context.add(nodeArray[from].compiledWord + '→' + wordBeforeJ.compiledWord + '@' + nodeArray[to].compiledWord);
        context.add(wordBeforeI.label + '@' + nodeArray[from].label + '→' + nodeArray[to].label);
        context.add(nodeArray[from].label + '→' + wordBeforeJ.label + '@' + nodeArray[to].label);
        List<Pair<String, Double>> pairList = model.predict(context.toArray(new String[0]));
        Pair<String, Double> maxPair = new Pair<String, Double>("null", -1.0);
//        System.out.println(context);
//        System.out.println(pairList);
        for (Pair<String, Double> pair : pairList)
        {
            if (pair.getValue() > maxPair.getValue() && !"null".equals(pair.getKey()))
            {
                maxPair = pair;
            }
        }
//        System.out.println(nodeArray[from].word + "→" + nodeArray[to].word + " : " + maxPair);

        return new Edge(from, to, maxPair.getKey(), (float) - Math.log(maxPair.getValue()));
    }
}
