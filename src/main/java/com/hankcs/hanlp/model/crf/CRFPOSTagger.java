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

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author hankcs
 */
public class CRFPOSTagger extends CRFTagger
{
    public CRFPOSTagger()
    {
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
            StringBuilder sbLine = new StringBuilder();
            String curWord = words[i];
            writeCell(curWord, sbLine, true);  // 0

            int length = curWord.length();
            // prefix
            writeCell(curWord.substring(0, 1), sbLine, true); // 1
            writeCell(length > 1 ? curWord.substring(0, 2) : "<>", sbLine, true); // 2
//            writeCell(length > 2 ? curWord.substring(0, 3) : "<>", sbLine, true); // 3


            // sufﬁx(w0, i)(i = 1, 2, 3)
            writeCell(curWord.substring(length - 1), sbLine, true); // 4
            writeCell(length > 1 ? curWord.substring(length - 2) : "<>", sbLine, true); // 5
//            writeCell(length > 2 ? curWord.substring(length - 3) : "<>", sbLine, true); // 6
            writeCell(iterator.next().label, sbLine, false);
            bw.write(sbLine.toString());
            bw.newLine();
        }
    }

    private void writeCell(String cell, StringBuilder sb, boolean tab) throws IOException
    {
        sb.append(cell);
        if (tab) sb.append("\t");
    }

    @Override
    protected String getFeatureTemplate()
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

    public void tag(List<String> wordList)
    {

    }
}
