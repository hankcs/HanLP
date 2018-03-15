package com.hankcs.hanlp.model.perceptron.model;

import com.hankcs.hanlp.model.perceptron.CWSTrainer;
import com.hankcs.hanlp.model.perceptron.PerceptronTrainer;
import junit.framework.TestCase;

import static java.lang.System.out;

public class LinearModelTest extends TestCase
{
    public static final String MODEL_FILE = "data/pku_mini.bin";

//    public void testLoad() throws Exception
//    {
//        LinearModel model = new LinearModel(MODEL_FILE);
//        PerceptronTrainer trainer = new CWSTrainer();
//        double[] prf = trainer.evaluate("icwb2-data/mini/pku_development.txt",
//                                                              model
//        );
//        out.printf("Performance - P:%.2f R:%.2f F:%.2f\n", prf[0], prf[1], prf[2]);
//    }
}