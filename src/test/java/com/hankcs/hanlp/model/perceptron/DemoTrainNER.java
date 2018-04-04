/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-28 15:46</create-date>
 *
 * <copyright file="DemoTrainNER.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

import java.io.IOException;

/**
 * @author hankcs
 */
public class DemoTrainNER
{
    public static void main(String[] args) throws IOException
    {
        PerceptronTrainer trainer = new NERTrainer();
        trainer.train("data/test/pku98/199801.txt", Config.NER_MODEL_FILE);
    }

    public static void trainYourNER()
    {
        PerceptronTrainer trainer = new NERTrainer()
        {
            @Override
            protected TagSet createTagSet()
            {
                NERTagSet tagSet = new NERTagSet();
                tagSet.nerLabels.add("YourNER1");
                tagSet.nerLabels.add("YourNER2");
                tagSet.nerLabels.add("YourNER3");
                return tagSet;
            }
        };
    }
}
