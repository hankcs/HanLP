/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/17 14:39</create-date>
 *
 * <copyright file="NSDictionaryMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary;

import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.NT;
import com.hankcs.hanlp.corpus.util.Precompiler;
import com.hankcs.hanlp.utility.Predefine;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * @author hankcs
 */
public class NTDictionaryMaker extends CommonDictionaryMaker
{
    TFDictionary tfDictionary = new TFDictionary();

    public NTDictionaryMaker(EasyDictionary dictionary)
    {
        super(dictionary);
    }

    @Override
    protected void addToDictionary(List<List<IWord>> sentenceList)
    {
//        logger.warning("开始制作词典");
        // 将非A的词语保存下来
        for (List<IWord> wordList : sentenceList)
        {
            for (IWord word : wordList)
            {
                if (!word.getLabel().equals(NT.Z.toString()))
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
        int i = 0;
        for (List<IWord> wordList : sentenceList)
        {
            Precompiler.compileWithoutNT(wordList);
            if (verbose)
            {
                System.out.print(++i + " / " + sentenceList.size() + " ");
                System.out.println("原始语料 " + wordList);
            }
            LinkedList<IWord> wordLinkedList = (LinkedList<IWord>) wordList;
            wordLinkedList.addFirst(new Word(Predefine.TAG_BIGIN, "S"));
            wordLinkedList.addLast(new Word(Predefine.TAG_END, "Z"));
            if (verbose) System.out.println("添加首尾 " + wordList);
            // 标注上文
            Iterator<IWord> iterator = wordLinkedList.iterator();
            IWord pre = iterator.next();
            while (iterator.hasNext())
            {
                IWord current = iterator.next();
                if (current.getLabel().startsWith("nt") && !pre.getLabel().startsWith("nt"))
                {
                    pre.setLabel(NT.A.toString());
                }
                pre = current;
            }
            if (verbose) System.out.println("标注上文 " + wordList);
            // 标注下文
            iterator = wordLinkedList.descendingIterator();
            pre = iterator.next();
            while (iterator.hasNext())
            {
                IWord current = iterator.next();
                if (current.getLabel().startsWith("nt") && !pre.getLabel().startsWith("nt"))
                {
                    pre.setLabel(NT.B.toString());
                }
                pre = current;
            }
            if (verbose) System.out.println("标注下文 " + wordList);
            // 标注中间
            {
                iterator = wordLinkedList.iterator();
                IWord first = iterator.next();
                IWord second = iterator.next();
                while (iterator.hasNext())
                {
                    IWord third = iterator.next();
                    if (first.getLabel().startsWith("nt") && third.getLabel().startsWith("nt") && !second.getLabel().startsWith("nt"))
                    {
                        second.setLabel(NT.X.toString());
                    }
                    first = second;
                    second = third;
                }
                if (verbose) System.out.println("标注中间 " + wordList);
            }
            // 处理整个
            ListIterator<IWord> listIterator = wordLinkedList.listIterator();
            while (listIterator.hasNext())
            {
                IWord word = listIterator.next();
                String label = word.getLabel();
                if (label.equals(label.toUpperCase())) continue;
                if (label.startsWith("nt"))
                {
                    StringBuilder sbPattern = new StringBuilder();
                    // 复杂机构
                    if (word instanceof CompoundWord)
                    {
                        listIterator.remove();
                        Word last = null;
                        for (Word inner : ((CompoundWord) word).innerList)
                        {
                            last = inner;
                            String innerLabel = inner.label;
                            if (innerLabel.startsWith("ns"))
                            {
                                inner.setValue(Predefine.TAG_PLACE);
                                inner.setLabel(NT.G.toString());
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if (innerLabel.startsWith("nt"))
                            {
                                inner.value = Predefine.TAG_GROUP;
                                inner.label = NT.K.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if (innerLabel.equals("b") || innerLabel.equals("ng") || innerLabel.equals("j"))
                            {
                                inner.label = NT.J.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if ("n".equals(innerLabel) ||
                                    "an".equals(innerLabel) ||
                                    "a".equals(innerLabel) ||
                                    "vn".equals(innerLabel) ||
                                    "vd".equals(innerLabel) ||
                                    "vl".equals(innerLabel) ||
                                    "v".equals(innerLabel) ||
                                    "vi".equals(innerLabel) ||
                                    "nnt".equals(innerLabel) ||
                                    "nnd".equals(innerLabel) ||
                                    "nf".equals(innerLabel) ||
                                    "cc".equals(innerLabel) ||
                                    "t".equals(innerLabel) ||
                                    "z".equals(innerLabel)
                                    )
                            {
                                inner.label = NT.C.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if ("nz".equals(innerLabel))
                            {
                                inner.label = NT.I.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if ("m".equals(innerLabel))
                            {
                                inner.value = Predefine.TAG_NUMBER;
                                inner.label = NT.M.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if ("w".equals(innerLabel))
                            {
                                inner.label = NT.W.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if (innerLabel.startsWith("nr") || "x".equals(innerLabel) || "nx".equals(innerLabel))
                            {
                                inner.value = Predefine.TAG_PEOPLE;
                                inner.label = NT.F.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if (innerLabel.startsWith("ni"))
                            {
                                inner.label = NT.D.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else if ("f".equals(innerLabel) || "s".equals(innerLabel))
                            {
                                inner.label = NT.L.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                            else
                            {
                                inner.label = NT.P.toString();
                                listIterator.add(inner);
                                sbPattern.append(inner.label);
                            }
                    }
                    if (last != null)
                    {
                        last.label = NT.D.toString();
                        sbPattern.deleteCharAt(sbPattern.length() - 1);
                        sbPattern.append(last.label);
                        tfDictionary.add(sbPattern.toString());
                        sbPattern.setLength(0);
                    }
                }
                else
                {
                    word.setLabel(NT.K.toString());
                }
            }
            else
            {
                word.setLabel(NT.Z.toString());
            }
        }
        if (verbose) System.out.println("处理整个 " + wordList);
        wordLinkedList.getFirst().setLabel(NT.S.toString());
    }

}

    @Override
    public boolean saveTxtTo(String path)
    {
        if (!super.saveTxtTo(path)) return false;
        return tfDictionary.saveKeyTo(path + ".pattern.txt");
    }
}
