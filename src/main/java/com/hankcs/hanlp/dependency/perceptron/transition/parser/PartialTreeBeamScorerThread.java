/**
 * Copyright 2014, Yahoo! Inc. and Mohammad Sadegh Rasooli
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.transition.features.FeatureExtractor;
import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.BeamElement;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Instance;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.State;

import java.util.ArrayList;
import java.util.concurrent.Callable;


public class PartialTreeBeamScorerThread implements Callable<ArrayList<BeamElement>>
{

    boolean isDecode;
    AveragedPerceptron classifier;
    Configuration configuration;
    Instance instance;
    ArrayList<Integer> dependencyRelations;
    int featureLength;
    int b;

    public PartialTreeBeamScorerThread(boolean isDecode, AveragedPerceptron classifier, Instance instance, Configuration configuration, ArrayList<Integer> dependencyRelations, int featureLength, int b)
    {
        this.isDecode = isDecode;
        this.classifier = classifier;
        this.configuration = configuration;
        this.instance = instance;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.b = b;
    }


    public ArrayList<BeamElement> call() throws Exception
    {
        ArrayList<BeamElement> elements = new ArrayList<BeamElement>(dependencyRelations.size() * 2 + 3);

        boolean isNonProjective = false;
        if (instance.isNonprojective())
        {
            isNonProjective = true;
        }

        State currentState = configuration.state;
        float prevScore = configuration.score;

        boolean canShift = ArcEager.canDo(Action.Shift, currentState);
        boolean canReduce = ArcEager.canDo(Action.Reduce, currentState);
        boolean canRightArc = ArcEager.canDo(Action.RightArc, currentState);
        boolean canLeftArc = ArcEager.canDo(Action.LeftArc, currentState);
        Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

        if (canShift)
        {
            if (isNonProjective || instance.actionCost(Action.Shift, -1, currentState) == 0)
            {
                float score = classifier.shiftScore(features, isDecode);
                float addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 0, -1));
            }
        }
        if (canReduce)
        {
            if (isNonProjective || instance.actionCost(Action.Reduce, -1, currentState) == 0)
            {
                float score = classifier.reduceScore(features, isDecode);
                float addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 1, -1));
            }

        }

        if (canRightArc)
        {
            float[] rightArcScores = classifier.rightArcScores(features, isDecode);
            for (int dependency : dependencyRelations)
            {
                if (isNonProjective || instance.actionCost(Action.RightArc, dependency, currentState) == 0)
                {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 2, dependency));
                }
            }
        }
        if (canLeftArc)
        {
            float[] leftArcScores = classifier.leftArcScores(features, isDecode);
            for (int dependency : dependencyRelations)
            {
                if (isNonProjective || instance.actionCost(Action.LeftArc, dependency, currentState) == 0)
                {
                    float score = leftArcScores[dependency];
                    float addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 3, dependency));

                }
            }
        }

        if (elements.size() == 0)
        {
            addAvailableBeamElements(elements, prevScore, canShift, canReduce, canRightArc, canLeftArc, features, classifier, isDecode, b, dependencyRelations);
        }

        return elements;
    }

    public static void addAvailableBeamElements(ArrayList<BeamElement> elements, float prevScore, boolean canShift, boolean canReduce, boolean canRightArc, boolean canLeftArc, Object[] features, AveragedPerceptron classifier, boolean isDecode, int b, ArrayList<Integer> dependencyRelations)
    {
        if (canShift)
        {
            float score = classifier.shiftScore(features, isDecode);
            float addedScore = score + prevScore;
            elements.add(new BeamElement(addedScore, b, 0, -1));
        }
        if (canReduce)
        {
            float score = classifier.reduceScore(features, isDecode);
            float addedScore = score + prevScore;
            elements.add(new BeamElement(addedScore, b, 1, -1));
        }

        if (canRightArc)
        {
            float[] rightArcScores = classifier.rightArcScores(features, isDecode);
            for (int dependency : dependencyRelations)
            {
                float score = rightArcScores[dependency];
                float addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 2, dependency));
            }
        }
        if (canLeftArc)
        {
            float[] leftArcScores = classifier.leftArcScores(features, isDecode);
            for (int dependency : dependencyRelations)
            {
                float score = leftArcScores[dependency];
                float addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 3, dependency));
            }
        }
    }
}