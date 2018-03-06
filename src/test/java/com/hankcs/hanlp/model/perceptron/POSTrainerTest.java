package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

import java.util.Arrays;

public class POSTrainerTest extends TestCase
{

    public void testTrain() throws Exception
    {
        PerceptronTrainer trainer = new POSTrainer();
        trainer.train("data/test/pku98/199801.txt", Config.POS_MODEL_FILE);
    }

    public void testLoad() throws Exception
    {
        PerceptronPOSTagger tagger = new PerceptronPOSTagger(Config.POS_MODEL_FILE);
        System.out.println(Arrays.toString(tagger.tag("中国 交响乐团 谭利华 在 布达拉宫 广场 演出".split(" "))));
    }
}