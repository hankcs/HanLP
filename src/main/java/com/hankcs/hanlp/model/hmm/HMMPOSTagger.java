/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-02 8:49 PM</create-date>
 *
 * <copyright file="HMMPOSTagger.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.tagset.POSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.tokenizer.lexical.POSTagger;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hankcs
 */
public class HMMPOSTagger extends HMMTrainer implements POSTagger
{
    POSTagSet tagSet;

    public HMMPOSTagger(HiddenMarkovModel model)
    {
        super(model);
        tagSet = new POSTagSet();
    }

    public HMMPOSTagger()
    {
        super();
        tagSet = new POSTagSet();
    }

    @Override
    protected List<String[]> convertToSequence(Sentence sentence)
    {
        List<Word> wordList = sentence.toSimpleWordList();
        List<String[]> xyList = new ArrayList<String[]>(wordList.size());
        for (Word word : wordList)
        {
            xyList.add(new String[]{word.getValue(), word.getLabel()});
        }
        return xyList;
    }

    @Override
    protected TagSet getTagSet()
    {
        return tagSet;
    }

    @Override
    public String[] tag(String... words)
    {
        int[] obsArray = new int[words.length];
        for (int i = 0; i < obsArray.length; i++)
        {
            obsArray[i] = vocabulary.idOf(words[i]);
        }
        int[] tagArray = new int[obsArray.length];
        model.predict(obsArray, tagArray);
        String[] tags = new String[obsArray.length];
        for (int i = 0; i < tagArray.length; i++)
        {
            tags[i] = tagSet.stringOf(tagArray[i]);
        }

        return tags;
    }

    @Override
    public String[] tag(List<String> wordList)
    {
        return tag(wordList.toArray(new String[0]));
    }
}
