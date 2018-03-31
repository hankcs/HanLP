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

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.model.crf.crfpp.TaggerImpl;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.List;

/**
 * @author hankcs
 */
public class CRFNERecognizer extends CRFTagger implements NERecognizer
{

    private NERTagSet tagSet;

    public CRFNERecognizer() throws IOException
    {
        this(HanLP.Config.CRFNERModelPath);
    }

    public CRFNERecognizer(String modelPath) throws IOException
    {
        super(modelPath);
        tagSet = new NERTagSet();
        if (model != null)
        {
            for (String y : model.getFeatureIndex_().getY_())
            {
                String label = NERTagSet.posOf(y);
                if (label.length() != y.length())
                    tagSet.nerLabels.add(label);
            }
        }
        else
        {
            addDefaultNERLabels();
        }
    }

    protected void addDefaultNERLabels()
    {
        tagSet.nerLabels.add("nr");
        tagSet.nerLabels.add("ns");
        tagSet.nerLabels.add("nt");
    }

    @Override
    protected void convertCorpus(Sentence sentence, BufferedWriter bw) throws IOException
    {
        List<String[]> collector = Utility.convertSentenceToNER(sentence, tagSet);
        for (String[] tuple : collector)
        {
            bw.write(tuple[0]);
            bw.write('\t');
            bw.write(tuple[1]);
            bw.write('\t');
            bw.write(tuple[2]);
            bw.newLine();
        }
    }

    @Override
    public String[] recognize(String[] wordArray, String[] posArray)
    {
        TaggerImpl tagger = createTagger();
        for (int i = 0; i < wordArray.length; i++)
        {
            tagger.add(new String[]{wordArray[i], posArray[i]});
        }
        tagger.parse();

        String[] tagArray = new String[wordArray.length];
        for (int i = 0; i < tagArray.length; i++)
        {
            tagArray[i] = tagger.yname(tagger.y(i));
        }

        return tagArray;
    }

    @Override
    public NERTagSet getNERTagSet()
    {
        return tagSet;
    }

    @Override
    protected String getDefaultFeatureTemplate()
    {
        return "# Unigram\n" +
            // form
            "U0:%x[-2,0]\n" +
            "U1:%x[-1,0]\n" +
            "U2:%x[0,0]\n" +
            "U3:%x[1,0]\n" +
            "U4:%x[2,0]\n" +
            // pos
            "U5:%x[-2,1]\n" +
            "U6:%x[-1,1]\n" +
            "U7:%x[0,1]\n" +
            "U8:%x[1,1]\n" +
            "U9:%x[2,1]\n" +
            // pos 2-gram
            "UA:%x[-2,1]%x[-1,1]\n" +
            "UB:%x[-1,1]%x[0,1]\n" +
            "UC:%x[0,1]%x[1,1]\n" +
            "UD:%x[1,1]%x[2,1]\n" +
            "UE:%x[2,1]%x[3,1]\n" +
            "\n" +
            "# Bigram\n" +
            "B";
    }
}
