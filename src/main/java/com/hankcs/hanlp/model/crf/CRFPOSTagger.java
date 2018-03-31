/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 上午3:04</create-date>
 *
 * <copyright file="CRFPOSTagger.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.crf.crfpp.TaggerImpl;
import com.hankcs.hanlp.tokenizer.lexical.POSTagger;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author hankcs
 */
public class CRFPOSTagger extends CRFTagger implements POSTagger
{
    public CRFPOSTagger() throws IOException
    {
        this(HanLP.Config.CRFPOSModelPath);
    }

    public CRFPOSTagger(String modelPath) throws IOException
    {
        super(modelPath);
    }

    @Override
    protected void convertCorpus(Sentence sentence, BufferedWriter bw) throws IOException
    {
        List<Word> simpleWordList = sentence.toSimpleWordList();
        List<String> wordList = new ArrayList<String>(simpleWordList.size());
        for (Word word : simpleWordList)
        {
            wordList.add(word.value);
        }
        String[] words = wordList.toArray(new String[0]);
        Iterator<Word> iterator = simpleWordList.iterator();
        for (int i = 0; i < words.length; i++)
        {
            String curWord = words[i];
            String[] cells = createCells(true);
            extractFeature(curWord, cells);
            cells[5] = iterator.next().label;
            for (int j = 0; j < cells.length; j++)
            {
                bw.write(cells[j]);
                if (j != cells.length - 1)
                    bw.write('\t');
            }
            bw.newLine();
        }
    }

    private String[] createCells(boolean withTag)
    {
        return withTag ? new String[6] : new String[5];
    }

    private void extractFeature(String curWord, String[] cells)
    {
        int length = curWord.length();
        cells[0] = curWord;
        cells[1] = curWord.substring(0, 1);
        cells[2] = length > 1 ? curWord.substring(0, 2) : "<>";
        // length > 2 ? curWord.substring(0, 3) : "<>"
        cells[3] = curWord.substring(length - 1);
        cells[4] = length > 1 ? curWord.substring(length - 2) : "<>";
        // length > 2 ? curWord.substring(length - 3) : "<>"
    }

    @Override
    protected String getDefaultFeatureTemplate()
    {
        return "# Unigram\n" +
            "U0:%x[-1,0]\n" +
            "U1:%x[0,0]\n" +
            "U2:%x[1,0]\n" +
            "U3:%x[0,1]\n" +
            "U4:%x[0,2]\n" +
            "U5:%x[0,3]\n" +
            "U6:%x[0,4]\n" +
//            "U7:%x[0,5]\n" +
//            "U8:%x[0,6]\n" +
            "\n" +
            "# Bigram\n" +
            "B";
    }

    public String[] tag(List<String> wordList)
    {
        String[] tagArray = new String[wordList.size()];
        TaggerImpl tagger = createTagger();
        for (String word : wordList)
        {
            String[] cells = createCells(false);
            extractFeature(word, cells);
            tagger.add(cells);
        }
        tagger.parse();
        for (int i = 0; i < tagger.ysize(); i++)
        {
            tagArray[i] = tagger.yname(tagger.y(i));
        }
        return tagArray;
    }

    @Override
    public String[] tag(String... words)
    {
        String[] tagArray = new String[words.length];
        TaggerImpl tagger = createTagger();
        for (String word : words)
        {
            String[] cells = createCells(false);
            extractFeature(word, cells);
            tagger.add(cells);
        }
        tagger.parse();
        for (int i = 0; i < tagger.size(); i++)
        {
            tagArray[i] = tagger.yname(tagger.y(i));
        }
        return tagArray;
    }
}