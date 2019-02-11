/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.accessories;

import com.hankcs.hanlp.dependency.perceptron.transition.configuration.CompactTree;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Evaluator
{
    public static double[] evaluate(String testPath, String predictedPath, HashSet<String> puncTags) throws IOException
    {
        CoNLLReader goldReader = new CoNLLReader(testPath);
        CoNLLReader predictedReader = new CoNLLReader(predictedPath);

        ArrayList<CompactTree> goldConfiguration = goldReader.readStringData();
        ArrayList<CompactTree> predConfiguration = predictedReader.readStringData();

        float unlabMatch = 0f;
        float labMatch = 0f;
        int all = 0;

        float fullULabMatch = 0f;
        float fullLabMatch = 0f;
        int numTree = 0;

        for (int i = 0; i < predConfiguration.size(); i++)
        {
            HashMap<Integer, Pair<Integer, String>> goldDeps = goldConfiguration.get(i).goldDependencies;
            HashMap<Integer, Pair<Integer, String>> predDeps = predConfiguration.get(i).goldDependencies;

            ArrayList<String> goldTags = goldConfiguration.get(i).posTags;

            numTree++;
            boolean fullMatch = true;
            boolean fullUnlabMatch = true;
            for (int dep : goldDeps.keySet())
            {
                if (!puncTags.contains(goldTags.get(dep - 1).trim()))
                {
                    all++;
                    int gh = goldDeps.get(dep).first;
                    int ph = predDeps.get(dep).first;
                    String gl = goldDeps.get(dep).second.trim();
                    String pl = predDeps.get(dep).second.trim();

                    if (ph == gh)
                    {
                        unlabMatch++;

                        if (pl.equals(gl))
                            labMatch++;
                        else
                        {
                            fullMatch = false;
                        }
                    }
                    else
                    {
                        fullMatch = false;
                        fullUnlabMatch = false;
                    }
                }
            }

            if (fullMatch)
                fullLabMatch++;
            if (fullUnlabMatch)
                fullULabMatch++;
        }

//        DecimalFormat format = new DecimalFormat("##.00");
        double labeledAccuracy = 100.0 * labMatch / all;
        double unlabaledAccuracy = 100.0 * unlabMatch / all;
//        System.err.println("Labeled accuracy: " + format.format(labeledAccuracy));
//        System.err.println("Unlabeled accuracy:  " + format.format(unlabaledAccuracy));
        double labExact = 100.0 * fullLabMatch / numTree;
        double ulabExact = 100.0 * fullULabMatch / numTree;
//        System.err.println("Labeled exact match:  " + format.format(labExact));
//        System.err.println("Unlabeled exact match:  " + format.format(ulabExact) + " \n");
        return new double[]{unlabaledAccuracy, labeledAccuracy};
    }
}
