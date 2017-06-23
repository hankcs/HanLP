/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 14:46</create-date>
 *
 * <copyright file="NRDictionaryMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary;

import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.corpus.util.Precompiler;
import com.hankcs.hanlp.utility.Predefine;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * nr词典（词典+ngram转移+词性转移矩阵）制作工具
 * @author hankcs
 */
public class NRDictionaryMaker extends CommonDictionaryMaker
{

    public NRDictionaryMaker(EasyDictionary dictionary)
    {
        super(dictionary);
    }

    @Override
    protected void addToDictionary(List<List<IWord>> sentenceList)
    {
        logger.warning("开始制作词典");
        // 将非A的词语保存下来
        for (List<IWord> wordList : sentenceList)
        {
            for (IWord word : wordList)
            {
                if (!word.getLabel().equals(NR.A.toString()))
                {
                    dictionaryMaker.add(word);
                }
            }
        }
        // 制作NGram词典
        for (List<IWord> wordList : sentenceList)
        {
            IWord pre = null;
            for (IWord word : wordList)
            {
                if (pre != null)
                {
                    nGramDictionaryMaker.addPair(pre, word);
                }
                pre = word;
            }
        }
    }

    @Override
    protected void roleTag(List<List<IWord>> sentenceList)
    {
        logger.info("开始标注角色");
        int i = 0;
        for (List<IWord> wordList : sentenceList)
        {
            logger.info(++i + " / " + sentenceList.size());
            if (verbose) System.out.println("原始语料 " + wordList);
            // 先标注A和K
            IWord pre = new Word("##始##", "begin");
            ListIterator<IWord> listIterator = wordList.listIterator();
            while (listIterator.hasNext())
            {
                IWord word = listIterator.next();
                if (!word.getLabel().equals(Nature.nr.toString()))
                {
                    word.setLabel(NR.A.toString());
                }
                else
                {
                    if (!pre.getLabel().equals(Nature.nr.toString()))
                    {
                        pre.setLabel(NR.K.toString());
                    }
                }
                pre = word;
            }
            if (verbose) System.out.println("标注非前 " + wordList);
            // 然后标注LM
            IWord next = new Word("##末##", "end");
            while (listIterator.hasPrevious())
            {
                IWord word = listIterator.previous();
                if (word.getLabel().equals(Nature.nr.toString()))
                {
                    String label = next.getLabel();
                    if (label.equals("A")) next.setLabel("L");
                    else if (label.equals("K")) next.setLabel("M");
                }
                next = word;
            }
            if (verbose) System.out.println("标注中后 " + wordList);
            // 拆分名字
            listIterator = wordList.listIterator();
            while (listIterator.hasNext())
            {
                IWord word = listIterator.next();
                if (word.getLabel().equals(Nature.nr.toString()))
                {
                    switch (word.getValue().length())
                    {
                        case 2:
                            if (word.getValue().startsWith("大")
                                    || word.getValue().startsWith("老")
                                    || word.getValue().startsWith("小")
                                    )
                            {
                                listIterator.add(new Word(word.getValue().substring(1, 2), NR.B.toString()));
                                word.setValue(word.getValue().substring(0, 1));
                                word.setLabel(NR.F.toString());
                            }
                            else if (word.getValue().endsWith("哥")
                                    || word.getValue().endsWith("公")
                                    || word.getValue().endsWith("姐")
                                    || word.getValue().endsWith("老")
                                    || word.getValue().endsWith("某")
                                    || word.getValue().endsWith("嫂")
                                    || word.getValue().endsWith("氏")
                                    || word.getValue().endsWith("总")
                                    )

                            {
                                listIterator.add(new Word(word.getValue().substring(1, 2), NR.G.toString()));
                                word.setValue(word.getValue().substring(0, 1));
                                word.setLabel(NR.B.toString());
                            }
                            else
                            {
                                listIterator.add(new Word(word.getValue().substring(1, 2), NR.E.toString()));
                                word.setValue(word.getValue().substring(0, 1));
                                word.setLabel(NR.B.toString());
                            }
                            break;
                        case 3:
                            listIterator.add(new Word(word.getValue().substring(1, 2), NR.C.toString()));
                            listIterator.add(new Word(word.getValue().substring(2, 3), NR.D.toString()));
                            word.setValue(word.getValue().substring(0, 1));
                            word.setLabel(NR.B.toString());
                            break;
                    }
                }
            }
            if (verbose) System.out.println("姓名拆分 " + wordList);
            // 上文成词
            listIterator = wordList.listIterator();
            pre = new Word("##始##", "begin");
            while (listIterator.hasNext())
            {
                IWord word = listIterator.next();
                if (word.getLabel().equals(NR.B.toString()))
                {
                    String combine = pre.getValue() + word.getValue();
                    if (dictionary.contains(combine))
                    {
                        pre.setValue(combine);
                        pre.setLabel("U");
                        listIterator.remove();
                    }
                }
                pre = word;
            }
            if (verbose) System.out.println("上文成词 " + wordList);
            // 头部成词
            next = new Word("##末##", "end");
            while (listIterator.hasPrevious())
            {
                IWord word = listIterator.previous();
                if (word.getLabel().equals(NR.B.toString()))
                {
                    String combine = word.getValue() + next.getValue();
                    if (dictionary.contains(combine))
                    {
                        next.setValue(combine);
                        next.setLabel(next.getLabel().equals(NR.C.toString()) ? NR.X.toString() : NR.Y.toString());
                        listIterator.remove();
                    }
                }
                next = word;
            }
            if (verbose) System.out.println("头部成词 " + wordList);
            // 尾部成词
            pre = new Word("##始##", "begin");
            while (listIterator.hasNext())
            {
                IWord word = listIterator.next();
                if (word.getLabel().equals(NR.D.toString()))
                {
                    String combine = pre.getValue() + word.getValue();
                    if (dictionary.contains(combine))
                    {
                        pre.setValue(combine);
                        pre.setLabel(NR.Z.toString());
                        listIterator.remove();
                    }
                }
                pre = word;
            }
            if (verbose) System.out.println("尾部成词 " + wordList);
            // 下文成词
            next = new Word("##末##", "end");
            while (listIterator.hasPrevious())
            {
                IWord word = listIterator.previous();
                if (word.getLabel().equals(NR.D.toString()))
                {
                    String combine = word.getValue() + next.getValue();
                    if (dictionary.contains(combine))
                    {
                        next.setValue(combine);
                        next.setLabel(NR.V.toString());
                        listIterator.remove();
                    }
                }
                next = word;
            }
            if (verbose) System.out.println("头部成词 " + wordList);
            LinkedList<IWord> wordLinkedList = (LinkedList<IWord>) wordList;
            wordLinkedList.addFirst(new Word(Predefine.TAG_BIGIN, "S"));
            wordLinkedList.addLast(new Word(Predefine.TAG_END, "A"));
            if (verbose) System.out.println("添加首尾 " + wordList);
        }
    }
}
