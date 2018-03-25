package com.hankcs.hanlp.algorithm;

import junit.framework.TestCase;

import static com.hankcs.hanlp.algorithm.ViterbiTest.Activity.clean;
import static com.hankcs.hanlp.algorithm.ViterbiTest.Activity.shop;
import static com.hankcs.hanlp.algorithm.ViterbiTest.Activity.walk;
import static com.hankcs.hanlp.algorithm.ViterbiTest.Weather.Rainy;
import static com.hankcs.hanlp.algorithm.ViterbiTest.Weather.Sunny;

public class ViterbiTest extends TestCase
{
    static enum Weather
    {
        Rainy,
        Sunny,
    }
    static enum Activity
    {
        walk,
        shop,
        clean,
    }
    static int[] states = new int[]{Rainy.ordinal(), Sunny.ordinal()};
    static int[] observations = new int[]{walk.ordinal(), shop.ordinal(), clean.ordinal()};
    double[] start_probability = new double[]{0.6, 0.4};
    double[][] transititon_probability = new double[][]{
        {0.7, 0.3},
        {0.4, 0.6},
    };
    double[][] emission_probability = new double[][]{
        {0.1, 0.4, 0.5},
        {0.6, 0.3, 0.1},
    };
    public void testCompute() throws Exception
    {
        for (int i = 0; i < start_probability.length; ++i)
        {
            start_probability[i] = -Math.log(start_probability[i]);
        }
        for (int i = 0; i < transititon_probability.length; ++i)
        {
            for (int j = 0; j < transititon_probability[i].length; ++j)
            {
                transititon_probability[i][j] = -Math.log(transititon_probability[i][j]);
            }
        }
        for (int i = 0; i < emission_probability.length; ++i)
        {
            for (int j = 0; j < emission_probability[i].length; ++j)
            {
                emission_probability[i][j] = -Math.log(emission_probability[i][j]);
            }
        }
        int[] result = Viterbi.compute(observations, states, start_probability, transititon_probability, emission_probability);
        for (int r : result)
        {
//            System.out.print(Weather.values()[r] + " ");
        }
//        System.out.println();
    }
}