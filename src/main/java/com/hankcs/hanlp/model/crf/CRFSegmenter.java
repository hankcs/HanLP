/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 上午1:07</create-date>
 *
 * <copyright file="CRFSegmenter.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.model.crf.crfpp.FeatureIndex;
import com.hankcs.hanlp.model.crf.crfpp.TaggerImpl;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.CWSInstance;
import com.hankcs.hanlp.tokenizer.lexical.Segmenter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
//import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;

/**
 * @author hankcs
 */
public class CRFSegmenter extends CRFTagger implements Segmenter
{

    private PerceptronSegmenter perceptronSegmenter;

    public CRFSegmenter() throws IOException
    {
        this(HanLP.Config.CRFCWSModelPath);
    }

    public CRFSegmenter(String modelPath) throws IOException
    {
        super(modelPath);
        if (modelPath != null)
        {
            perceptronSegmenter = new PerceptronSegmenter(this.model);
        }
    }

    @Override
    protected void convertCorpus(Sentence sentence, BufferedWriter bw) throws IOException
    {
        for (Word w : sentence.toSimpleWordList())
        {
            String word = CharTable.convert(w.value);
            if (word.length() == 1)
            {
                bw.write(word);
                bw.write('\t');
                bw.write('S');
                bw.write('\n');
            }
            else
            {
                bw.write(word.charAt(0));
                bw.write('\t');
                bw.write('B');
                bw.write('\n');
                for (int i = 1; i < word.length() - 1; ++i)
                {
                    bw.write(word.charAt(i));
                    bw.write('\t');
                    bw.write('M');
                    bw.write('\n');
                }
                bw.write(word.charAt(word.length() - 1));
                bw.write('\t');
                bw.write('E');
                bw.write('\n');
            }
        }
    }

    public List<String> segment(String text)
    {
        List<String> wordList = new LinkedList<String>();
        segment(text, CharTable.convert(text), wordList);

        return wordList;
    }

    @Override
    public void segment(String text, String normalized, List<String> wordList)
    {
        perceptronSegmenter.segment(text, createInstance(normalized), wordList);
    }

    private CWSInstance createInstance(String text)
    {
        final FeatureTemplate[] featureTemplateArray = model.getFeatureTemplateArray();
        return new CWSInstance(text, model.featureMap)
        {
            @Override
            protected int[] extractFeature(String sentence, FeatureMap featureMap, int position)
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
                        int offset = offsetIterator.next()[0] + position;
                        if (offset < 0)
                            sbFeature.append(FeatureIndex.BOS[-(offset + 1)]);
                        else if (offset >= sentence.length())
                            sbFeature.append(FeatureIndex.EOS[offset - sentence.length()]);
                        else
                            sbFeature.append(sentence.charAt(offset));
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
            "U0:%x[-1,0]\n" +
            "U1:%x[0,0]\n" +
            "U2:%x[1,0]\n" +
            "U3:%x[-2,0]%x[-1,0]\n" +
            "U4:%x[-1,0]%x[0,0]\n" +
            "U5:%x[0,0]%x[1,0]\n" +
            "U6:%x[1,0]%x[2,0]\n" +
            "\n" +
            "# Bigram\n" +
            "B";
    }
}
