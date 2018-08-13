/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-02 9:15 PM</create-date>
 *
 * <copyright file="HMMNERecognizer.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.util.List;

/**
 * @author hankcs
 */
public class HMMNERecognizer extends HMMTrainer implements NERecognizer
{
    NERTagSet tagSet;

    public HMMNERecognizer(HiddenMarkovModel model)
    {
        super(model);
        tagSet = new NERTagSet();
        tagSet.nerLabels.add("nr");
        tagSet.nerLabels.add("ns");
        tagSet.nerLabels.add("nt");
    }

    public HMMNERecognizer()
    {
        this(new FirstOrderHiddenMarkovModel());
    }

    @Override
    protected List<String[]> convertToSequence(Sentence sentence)
    {
        List<String[]> collector = Utility.convertSentenceToNER(sentence, tagSet);
        for (String[] pair : collector)
        {
            pair[1] = pair[2];
        }

        return collector;
    }

    @Override
    protected TagSet getTagSet()
    {
        return tagSet;
    }

    @Override
    public String[] recognize(String[] wordArray, String[] posArray)
    {
        int[] obsArray = new int[wordArray.length];
        for (int i = 0; i < obsArray.length; i++)
        {
            obsArray[i] = vocabulary.idOf(wordArray[i]);
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
    public NERTagSet getNERTagSet()
    {
        return tagSet;
    }
}
