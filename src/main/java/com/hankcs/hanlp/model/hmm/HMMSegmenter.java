/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-13 2:05 PM</create-date>
 *
 * <copyright file="HMMSegmenter.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.model.perceptron.tagset.CWSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.lexical.Segmenter;

import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class HMMSegmenter extends HMMTrainer implements Segmenter
{
    CWSTagSet tagSet;

    public HMMSegmenter(HiddenMarkovModel model)
    {
        super(model);
        tagSet = new CWSTagSet();
    }

    public HMMSegmenter()
    {
        tagSet = new CWSTagSet();
    }

    @Override
    public List<String> segment(String text)
    {
        List<String> wordList = new LinkedList<String>();
        segment(text, CharTable.convert(text), wordList);
        return wordList;
    }

    @Override
    public void segment(String text, String normalized, List<String> output)
    {
        int[] obsArray = new int[text.length()];
        for (int i = 0; i < obsArray.length; i++)
        {
            obsArray[i] = vocabulary.idOf(normalized.substring(i, i + 1));
        }
        int[] tagArray = new int[text.length()];
        model.predict(obsArray, tagArray);
        StringBuilder result = new StringBuilder();
        result.append(text.charAt(0));

        for (int i = 1; i < tagArray.length; i++)
        {
            if (tagArray[i] == tagSet.B || tagArray[i] == tagSet.S)
            {
                output.add(result.toString());
                result.setLength(0);
            }
            result.append(text.charAt(i));
        }
        if (result.length() != 0)
        {
            output.add(result.toString());
        }
    }

    @Override
    protected List<String[]> convertToSequence(Sentence sentence)
    {
        List<String[]> charList = new LinkedList<String[]>();
        for (Word w : sentence.toSimpleWordList())
        {
            String word = CharTable.convert(w.value);
            if (word.length() == 1)
            {
                charList.add(new String[]{word, "S"});
            }
            else
            {
                charList.add(new String[]{word.substring(0, 1), "B"});
                for (int i = 1; i < word.length() - 1; ++i)
                {
                    charList.add(new String[]{word.substring(i, i + 1), "M"});
                }
                charList.add(new String[]{word.substring(word.length() - 1), "E"});
            }
        }
        return charList;
    }

    @Override
    protected TagSet getTagSet()
    {
        return tagSet;
    }

    /**
     * 获取兼容旧的Segment接口
     *
     * @return
     */
    public Segment toSegment()
    {
        return new Segment()
        {
            @Override
            protected List<Term> segSentence(char[] sentence)
            {
                List<String> wordList = segment(new String(sentence));
                List<Term> termList = new LinkedList<Term>();
                for (String word : wordList)
                {
                    termList.add(new Term(word, null));
                }
                return termList;
            }
        }.enableCustomDictionary(false);
    }
}