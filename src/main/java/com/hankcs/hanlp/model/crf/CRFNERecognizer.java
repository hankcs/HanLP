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
import com.hankcs.hanlp.model.crf.crfpp.FeatureIndex;
import com.hankcs.hanlp.model.crf.crfpp.TaggerImpl;
import com.hankcs.hanlp.model.perceptron.PerceptronNERecognizer;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.NERInstance;
import com.hankcs.hanlp.model.perceptron.instance.POSInstance;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class CRFNERecognizer extends CRFTagger implements NERecognizer
{
    private NERTagSet tagSet;
    /**
     * 复用感知机的解码模块
     */
    private PerceptronNERecognizer perceptronNERecognizer;

    public CRFNERecognizer() throws IOException
    {
        this(HanLP.Config.CRFNERModelPath);
    }

    public CRFNERecognizer(String modelPath) throws IOException
    {
        this(modelPath,null);
    }

    public CRFNERecognizer(String modelPath,String[] customNERTags) throws IOException
    {
        super(modelPath);
        if (model == null)
        {
            tagSet = new NERTagSet();
            addDefaultNERLabels();
            if (customNERTags != null) {
                for (String nerTags : customNERTags) {
                    addNERLabels(nerTags);
                }
            }
        }
        else
        {
            perceptronNERecognizer = new PerceptronNERecognizer(this.model);
            tagSet = perceptronNERecognizer.getNERTagSet();
        }
    }

    protected void addDefaultNERLabels()
    {
        tagSet.nerLabels.add("nr");
        tagSet.nerLabels.add("ns");
        tagSet.nerLabels.add("nt");
    }

    public void addNERLabels(String newNerTag)
    {
        tagSet.nerLabels.add(newNerTag);
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
        return perceptronNERecognizer.recognize(createInstance(wordArray, posArray));
    }

    @Override
    public NERTagSet getNERTagSet()
    {
        return tagSet;
    }

    private NERInstance createInstance(String[] wordArray, String[] posArray)
    {
        final FeatureTemplate[] featureTemplateArray = model.getFeatureTemplateArray();
        return new NERInstance(wordArray, posArray, model.featureMap)
        {
            @Override
            protected int[] extractFeature(String[] wordArray, String[] posArray, FeatureMap featureMap, int position)
            {
                StringBuilder sbFeature = new StringBuilder();
                List<Integer> featureVec = new LinkedList<Integer>();
                for (int i = 0; i < featureTemplateArray.length; i++)
                {
                    Iterator<int[]> offsetIterator = featureTemplateArray[i].offsetList.iterator();
                    Iterator<String> delimiterIterator = featureTemplateArray[i].delimiterList.iterator();
                    delimiterIterator.next(); // ignore U0 之类的id
                    while (offsetIterator.hasNext())
                    {
                        int[] offset = offsetIterator.next();
                        int t = offset[0] + position;
                        boolean first = offset[1] == 0;
                        if (t < 0)
                            sbFeature.append(FeatureIndex.BOS[-(t + 1)]);
                        else if (t >= wordArray.length)
                            sbFeature.append(FeatureIndex.EOS[t - wordArray.length]);
                        else
                            sbFeature.append(first ? wordArray[t] : posArray[t]);
                        if (delimiterIterator.hasNext())
                            sbFeature.append(delimiterIterator.next());
                        else
                            sbFeature.append(i);
                    }
                    addFeatureThenClear(sbFeature, featureVec, featureMap);
                }
                return toFeatureArray(featureVec);
            }
        };
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
