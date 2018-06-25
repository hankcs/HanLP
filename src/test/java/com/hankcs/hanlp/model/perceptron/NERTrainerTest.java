package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

import java.util.Arrays;

public class NERTrainerTest extends TestCase
{
    public void testTrain() throws Exception
    {
        PerceptronTrainer trainer = new NERTrainer();
        trainer.train("data/test/pku98/199801.txt", Config.NER_MODEL_FILE);
    }

    public void testTag() throws Exception
    {
        PerceptronNERecognizer recognizer = new PerceptronNERecognizer(Config.NER_MODEL_FILE);
        System.out.println(Arrays.toString(recognizer.recognize("吴忠市 乳制品 公司 谭利华 来到 布达拉宫 广场".split(" "), "ns n n nr p ns n".split(" "))));
    }
}