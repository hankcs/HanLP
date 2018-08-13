/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-13 2:17 PM</create-date>
 *
 * <copyright file="HMMTrainer.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.model.perceptron.utility.IOUtility;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public abstract class HMMTrainer
{
    HiddenMarkovModel model;
    Vocabulary vocabulary;

    public HMMTrainer(HiddenMarkovModel model, Vocabulary vocabulary)
    {
        this.model = model;
        this.vocabulary = vocabulary;
    }

    public HMMTrainer(HiddenMarkovModel model)
    {
        this(model, new Vocabulary());
    }

    public HMMTrainer()
    {
        this(new FirstOrderHiddenMarkovModel());
    }

    public void train(String corpus) throws IOException
    {
        final List<List<String[]>> sequenceList = new LinkedList<List<String[]>>();
        IOUtility.loadInstance(corpus, new InstanceHandler()
        {
            @Override
            public boolean process(Sentence sentence)
            {
                sequenceList.add(convertToSequence(sentence));
                return false;
            }
        });

        TagSet tagSet = getTagSet();

        List<int[][]> sampleList = new ArrayList<int[][]>(sequenceList.size());
        for (List<String[]> sequence : sequenceList)
        {
            int[][] sample = new int[2][sequence.size()];
            int i = 0;
            for (String[] os : sequence)
            {
                sample[0][i] = vocabulary.idOf(os[0]);
                assert sample[0][i] != -1;
                sample[1][i] = tagSet.add(os[1]);
                assert sample[1][i] != -1;
                ++i;
            }
            sampleList.add(sample);
        }

        model.train(sampleList);
        vocabulary.mutable = false;
    }

    protected abstract List<String[]> convertToSequence(Sentence sentence);
    protected abstract TagSet getTagSet();
}
