/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-27 下午4:28</create-date>
 *
 * <copyright file="DemoTrainPOS.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import java.io.IOException;

/**
 * @author hankcs
 */
public class DemoTrainPOS
{
    public static void main(String[] args) throws IOException
    {
        PerceptronTrainer trainer = new POSTrainer();
        trainer.train("data/test/pku98/199801.txt", Config.POS_MODEL_FILE);
    }
}
