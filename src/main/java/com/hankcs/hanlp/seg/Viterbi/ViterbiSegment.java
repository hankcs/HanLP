/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/1/19 20:51</create-date>
 *
 * <copyright file="ViterbiSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.Viterbi;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.recognition.nr.JapanesePersonRecognition;
import com.hankcs.hanlp.recognition.nr.PersonRecognition;
import com.hankcs.hanlp.recognition.nr.TranslatedPersonRecognition;
import com.hankcs.hanlp.recognition.ns.PlaceRecognition;
import com.hankcs.hanlp.recognition.nt.OrganizationRecognition;
import com.hankcs.hanlp.seg.WordBasedSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.seg.common.WordNet;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * Viterbi分词器<br>
 * 也是最短路分词，最短路求解采用Viterbi算法
 *
 * @author hankcs
 */
public class ViterbiSegment extends WordBasedSegment
{
    private DoubleArrayTrie<CoreDictionary.Attribute> dat;

    public ViterbiSegment()
    {
        this.dat = CustomDictionary.dat;
    }

    /**
     * @param customPath 自定义字典路径（绝对路径，多词典使用英文分号隔开）
     */
    public ViterbiSegment(String customPath)
    {
        loadCustomDic(customPath, false);
    }

    /**
     * @param customPath customPath 自定义字典路径（绝对路径，多词典使用英文分号隔开）
     * @param cache      是否缓存词典
     */
    public ViterbiSegment(String customPath, boolean cache)
    {
        loadCustomDic(customPath, cache);
    }

    public DoubleArrayTrie<CoreDictionary.Attribute> getDat()
    {
        return dat;
    }

    public void setDat(DoubleArrayTrie<CoreDictionary.Attribute> dat)
    {
        this.dat = dat;
    }

    @Override
    protected List<Term> segSentence(char[] sentence)
    {
//        long start = System.currentTimeMillis();
        WordNet wordNetAll = new WordNet(sentence);
        ////////////////生成词网////////////////////
        generateWordNet(wordNetAll);
        ///////////////生成词图////////////////////
//        System.out.println("构图：" + (System.currentTimeMillis() - start));
        if (HanLP.Config.DEBUG)
        {
            System.out.printf("粗分词网：\n%s\n", wordNetAll);
        }
//        start = System.currentTimeMillis();
        List<Vertex> vertexList = viterbi(wordNetAll);
//        System.out.println("最短路：" + (System.currentTimeMillis() - start));

        if (config.useCustomDictionary)
        {
            if (config.indexMode > 0)
                combineByCustomDictionary(vertexList, this.dat, wordNetAll);
            else combineByCustomDictionary(vertexList, this.dat);
        }

        if (HanLP.Config.DEBUG)
        {
            System.out.println("粗分结果" + convert(vertexList, false));
        }

        // 数字识别
        if (config.numberQuantifierRecognize)
        {
            mergeNumberQuantifier(vertexList, wordNetAll, config);
        }

        // 实体命名识别
        if (config.ner)
        {
            WordNet wordNetOptimum = new WordNet(sentence, vertexList);
            int preSize = wordNetOptimum.size();
            if (config.nameRecognize)
            {
                PersonRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.translatedNameRecognize)
            {
                TranslatedPersonRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.japaneseNameRecognize)
            {
                JapanesePersonRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.placeRecognize)
            {
                PlaceRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (config.organizationRecognize)
            {
                // 层叠隐马模型——生成输出作为下一级隐马输入
                wordNetOptimum.clean();
                vertexList = viterbi(wordNetOptimum);
                wordNetOptimum.clear();
                wordNetOptimum.addAll(vertexList);
                preSize = wordNetOptimum.size();
                OrganizationRecognition.recognition(vertexList, wordNetOptimum, wordNetAll);
            }
            if (wordNetOptimum.size() != preSize)
            {
                vertexList = viterbi(wordNetOptimum);
                if (HanLP.Config.DEBUG)
                {
                    System.out.printf("细分词网：\n%s\n", wordNetOptimum);
                }
            }
        }

        // 如果是索引模式则全切分
        if (config.indexMode > 0)
        {
            return decorateResultForIndexMode(vertexList, wordNetAll);
        }

        // 是否标注词性
        if (config.speechTagging)
        {
            speechTagging(vertexList);
        }

        return convert(vertexList, config.offset);
    }

    private static List<Vertex> viterbi(WordNet wordNet)
    {
        // 避免生成对象，优化速度
        LinkedList<Vertex> nodes[] = wordNet.getVertexes();
        LinkedList<Vertex> vertexList = new LinkedList<Vertex>();
        for (Vertex node : nodes[1])
        {
            node.updateFrom(nodes[0].getFirst());
        }
        for (int i = 1; i < nodes.length - 1; ++i)
        {
            LinkedList<Vertex> nodeArray = nodes[i];
            if (nodeArray == null) continue;
            for (Vertex node : nodeArray)
            {
                if (node.from == null) continue;
                for (Vertex to : nodes[i + node.realWord.length()])
                {
                    to.updateFrom(node);
                }
            }
        }
        Vertex from = nodes[nodes.length - 1].getFirst();
        while (from != null)
        {
            vertexList.addFirst(from);
            from = from.from;
        }
        return vertexList;
    }

    private void loadCustomDic(String customPath, boolean isCache)
    {
        if (TextUtility.isBlank(customPath))
        {
            return;
        }
        logger.info("开始加载自定义词典:" + customPath);
        DoubleArrayTrie<CoreDictionary.Attribute> dat = new DoubleArrayTrie<CoreDictionary.Attribute>();
        String path[] = customPath.split(";");
        String mainPath = path[0];
        StringBuilder combinePath = new StringBuilder();
        for (String aPath : path)
        {
            combinePath.append(aPath.trim());
        }
        File file = new File(mainPath);
        mainPath = file.getParent() + "/" + Math.abs(combinePath.toString().hashCode());
        mainPath = mainPath.replace("\\", "/");
        if (CustomDictionary.loadMainDictionary(mainPath, path, dat, isCache))
        {
            this.setDat(dat);
        }
    }

    /**
     * 第二次维特比，可以利用前一次的结果，降低复杂度
     *
     * @param wordNet
     * @return
     */
//    private static List<Vertex> viterbiOptimal(WordNet wordNet)
//    {
//        LinkedList<Vertex> nodes[] = wordNet.getVertexes();
//        LinkedList<Vertex> vertexList = new LinkedList<Vertex>();
//        for (Vertex node : nodes[1])
//        {
//            if (node.isNew)
//                node.updateFrom(nodes[0].getFirst());
//        }
//        for (int i = 1; i < nodes.length - 1; ++i)
//        {
//            LinkedList<Vertex> nodeArray = nodes[i];
//            if (nodeArray == null) continue;
//            for (Vertex node : nodeArray)
//            {
//                if (node.from == null) continue;
//                if (node.isNew)
//                {
//                    for (Vertex to : nodes[i + node.realWord.length()])
//                    {
//                        to.updateFrom(node);
//                    }
//                }
//            }
//        }
//        Vertex from = nodes[nodes.length - 1].getFirst();
//        while (from != null)
//        {
//            vertexList.addFirst(from);
//            from = from.from;
//        }
//        return vertexList;
//    }
}
