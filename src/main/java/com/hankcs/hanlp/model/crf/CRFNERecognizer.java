/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 上午3:45</create-date>
 *
 * <copyright file="CRFNERecognizer.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * @author hankcs
 */
public class CRFNERecognizer extends CRFTagger
{
    @Override
    protected void convertCorpus(Sentence sentence, BufferedWriter bw) throws IOException
    {

        NERTagSet tagSet = new NERTagSet();
        tagSet.nerLabels.add("nr");
        tagSet.nerLabels.add("ns");
        tagSet.nerLabels.add("nt");
        List<String[]> collector = new LinkedList<String[]>();
        Set<String> nerLabels = tagSet.nerLabels;
        for (IWord word : sentence.wordList)
        {
            if (word instanceof CompoundWord)
            {
                List<Word> wordList = ((CompoundWord) word).innerList;
                Word[] words = wordList.toArray(new Word[0]);

                if (nerLabels.contains(word.getLabel()))
                {
                    collector.add(new String[]{words[0].value, words[0].label, tagSet.B_TAG_PREFIX + word.getLabel()});
                    for (int i = 1; i < words.length - 1; i++)
                    {
                        collector.add(new String[]{words[i].value, words[i].label, tagSet.M_TAG_PREFIX + word.getLabel()});
                    }
                    collector.add(new String[]{words[words.length - 1].value, words[words.length - 1].label,
                        tagSet.E_TAG_PREFIX + word.getLabel()});
                }
                else
                {
                    for (Word w : words)
                    {
                        collector.add(new String[]{w.value, w.label, tagSet.O_TAG});
                    }
                }
            }
            else
            {
                if (nerLabels.contains(word.getLabel()))
                {
                    // 单个实体
                    collector.add(new String[]{word.getValue(), word.getLabel(), tagSet.S_TAG});
                }
                else
                {
                    collector.add(new String[]{word.getValue(), word.getLabel(), tagSet.O_TAG});
                }
            }
        }
        String[] wordArray = new String[collector.size()];
        String[] posArray = new String[collector.size()];
        String[] tagArray = new String[collector.size()];
        int i = 0;
        for (String[] tuple : collector)
        {
            wordArray[i] = tuple[0];
            posArray[i] = tuple[1];
            tagArray[i] = tuple[2];
            ++i;
        }
    }

    @Override
    protected String getFeatureTemplate()
    {
        return null;
    }
}
