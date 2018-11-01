/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/2 20:54</create-date>
 *
 * <copyright file="NeuralNetworkDependencyParser.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.dependency.AbstractDependencyParser;
import com.hankcs.hanlp.dependency.IDependencyParser;
import com.hankcs.hanlp.dependency.nnparser.option.BasicOption;
import com.hankcs.hanlp.dependency.nnparser.option.ConfigOption;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;
import com.hankcs.hanlp.dependency.nnparser.util.PosTagUtil;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于神经网络分类模型arc-standard转移动作的判决式依存句法分析器
 * @author hankcs
 */
public class NeuralNetworkDependencyParser extends AbstractDependencyParser
{
    private parser_dll parser_dll;

    public NeuralNetworkDependencyParser(Segment segment)
    {
        super(segment);
        parser_dll = new parser_dll();
        setDeprelTranslater(ConfigOption.DEPRL_DESCRIPTION_PATH).enableDeprelTranslator(true);
    }

    public NeuralNetworkDependencyParser()
    {
        this(NLPTokenizer.ANALYZER);
    }

    @Override
    public CoNLLSentence parse(List<Term> termList)
    {
        List<String> posTagList = PosTagUtil.to863(termList);
        List<String> wordList = new ArrayList<String>(termList.size());
        for (Term term : termList)
        {
            wordList.add(term.word);
        }
        List<Integer> heads = new ArrayList<Integer>(termList.size());
        List<String> deprels = new ArrayList<String>(termList.size());
        parser_dll.parse(wordList, posTagList, heads, deprels);

        CoNLLWord[] wordArray = new CoNLLWord[termList.size()];
        for (int i = 0; i < wordArray.length; ++i)
        {
            wordArray[i] = new CoNLLWord(i + 1, wordList.get(i), posTagList.get(i), termList.get(i).nature.toString());
            wordArray[i].DEPREL = deprels.get(i);
        }
        for (int i = 0; i < wordArray.length; ++i)
        {
            int index = heads.get(i) - 1;
            if (index < 0)
            {
                wordArray[i].HEAD = CoNLLWord.ROOT;
                continue;
            }
            wordArray[i].HEAD = wordArray[index];
        }
        return new CoNLLSentence(wordArray);
    }

    /**
     * 分析句子的依存句法
     *
     * @param termList 句子，可以是任何具有词性标注功能的分词器的分词结果
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(List<Term> termList)
    {
        return new NeuralNetworkDependencyParser().parse(termList);
    }

    /**
     * 分析句子的依存句法
     *
     * @param sentence 句子
     * @return CoNLL格式的依存句法树
     */
    public static CoNLLSentence compute(String sentence)
    {
        return new NeuralNetworkDependencyParser().parse(sentence);
    }
}
