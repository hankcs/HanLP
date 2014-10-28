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
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.Predefine;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class NRDictionaryMaker extends CommonDictionaryMaker
{

    public NRDictionaryMaker(EasyDictionary dictionary)
    {
        super(dictionary);
    }

    @Override
    protected void addToDictionary(List<List<Word>> sentenceList)
    {
        logger.warning("开始制作词典");
        // 将非A的词语保存下来
        for (List<Word> wordList : sentenceList)
        {
            for (Word word : wordList)
            {
                if (!word.label.equals(NR.A.toString()))
                {
                    dictionaryMaker.add(word);
                }
            }
        }
        // 制作NGram词典
        for (List<Word> wordList : sentenceList)
        {
            Word pre = null;
            for (Word word : wordList)
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
    protected void roleTag(List<List<Word>> sentenceList)
    {
        logger.info("开始标注角色");
        int i = 0;
        for (List<Word> wordList : sentenceList)
        {
            logger.info(++i + " / " + sentenceList.size());
            if (verbose) System.out.println("原始语料 " + wordList);
            // 先标注A和K
            Word pre = new Word("##始##", "begin");
            ListIterator<Word> listIterator = wordList.listIterator();
            while (listIterator.hasNext())
            {
                Word word = listIterator.next();
                if (!word.label.equals(Nature.nr.toString()))
                {
                    word.label = NR.A.toString();
                }
                else
                {
                    if (!pre.label.equals(Nature.nr.toString()))
                    {
                        pre.label = NR.K.toString();
                    }
                }
                pre = word;
            }
            if (verbose) System.out.println("标注非前 " + wordList);
            // 然后标注LM
            Word next = new Word("##末##", "end");
            while (listIterator.hasPrevious())
            {
                Word word = listIterator.previous();
                if (word.label.equals(Nature.nr.toString()))
                {
                    switch (next.label)
                    {
                        case "A":
                            next.label = "L";
                            break;
                        case "K":
                            next.label = "M";
                            break;
                    }
                }
                next = word;
            }
            if (verbose) System.out.println("标注中后 " + wordList);
            // 拆分名字
            listIterator = wordList.listIterator();
            while (listIterator.hasNext())
            {
                Word word = listIterator.next();
                if (word.label.equals(Nature.nr.toString()))
                {
                    switch (word.value.length())
                    {
                        case 2:
                            if (word.value.startsWith("大")
                                    || word.value.startsWith("老")
                                    || word.value.startsWith("小")
                                    )
                            {
                                listIterator.add(new Word(word.value.substring(1, 2), NR.B.toString()));
                                word.value = word.value.substring(0, 1);
                                word.label = NR.F.toString();
                            }
                            else if (word.value.endsWith("哥")
                                    || word.value.endsWith("公")
                                    || word.value.endsWith("姐")
                                    || word.value.endsWith("老")
                                    || word.value.endsWith("某")
                                    || word.value.endsWith("嫂")
                                    || word.value.endsWith("氏")
                                    || word.value.endsWith("总")
                                    )

                            {
                                listIterator.add(new Word(word.value.substring(1, 2), NR.G.toString()));
                                word.value = word.value.substring(0, 1);
                                word.label = NR.B.toString();
                            }
                            else
                            {
                                listIterator.add(new Word(word.value.substring(1, 2), NR.E.toString()));
                                word.value = word.value.substring(0, 1);
                                word.label = NR.B.toString();
                            }
                            break;
                        case 3:
                            listIterator.add(new Word(word.value.substring(1, 2), NR.C.toString()));
                            listIterator.add(new Word(word.value.substring(2, 3), NR.D.toString()));
                            word.value = word.value.substring(0, 1);
                            word.label = NR.B.toString();
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
                Word word = listIterator.next();
                if (word.label.equals(NR.B.toString()))
                {
                    String combine = pre.value + word.value;
                    if (dictionary.contains(combine))
                    {
                        pre.value = combine;
                        pre.label = "U";
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
                Word word = listIterator.previous();
                if (word.label.equals(NR.B.toString()))
                {
                    String combine = word.value + next.value;
                    if (dictionary.contains(combine))
                    {
                        next.value = combine;
                        next.label = next.label.equals(NR.C.toString()) ? NR.X.toString() : NR.Y.toString();
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
                Word word = listIterator.next();
                if (word.label.equals(NR.D.toString()))
                {
                    String combine = pre.value + word.value;
                    if (dictionary.contains(combine))
                    {
                        pre.value = combine;
                        pre.label = NR.Z.toString();
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
                Word word = listIterator.previous();
                if (word.label.equals(NR.D.toString()))
                {
                    String combine = word.value + next.value;
                    if (dictionary.contains(combine))
                    {
                        next.value = combine;
                        next.label = NR.V.toString();
                        listIterator.remove();
                    }
                }
                next = word;
            }
            if (verbose) System.out.println("头部成词 " + wordList);
            LinkedList<Word> wordLinkedList = (LinkedList<Word>) wordList;
            wordLinkedList.addFirst(new Word(Predefine.TAG_BIGIN, "S"));
            wordLinkedList.addLast(new Word(Predefine.TAG_END, "A"));
            if (verbose) System.out.println("添加首尾 " + wordList);
        }
    }

    public static void main(String[] args)
    {
        EasyDictionary dictionary = EasyDictionary.create("data/dictionary/2014_dictionary.txt");
        final NRDictionaryMaker nrDictionaryMaker = new NRDictionaryMaker(dictionary);
        CorpusLoader.walk("data/corpus/2014/", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                nrDictionaryMaker.compute(document.getSimpleSentenceList());
            }
        });
        nrDictionaryMaker.saveTxtTo("D:\\JavaProjects\\HanLP\\data\\dictionary\\person\\nr1");
    }

}
