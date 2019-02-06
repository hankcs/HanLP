/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.transition.features.FeatureExtractor;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.BeamElement;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.State;

import java.util.ArrayList;
import java.util.concurrent.Callable;

import static com.hankcs.hanlp.dependency.perceptron.transition.parser.PartialTreeBeamScorerThread.addAvailableBeamElements;


public class BeamScorerThread implements Callable<ArrayList<BeamElement>>
{

    boolean isDecode;
    AveragedPerceptron classifier;
    Configuration configuration;
    ArrayList<Integer> dependencyRelations;
    int featureLength;
    int b;
    boolean rootFirst;

    public BeamScorerThread(boolean isDecode, AveragedPerceptron classifier, Configuration configuration, ArrayList<Integer> dependencyRelations, int featureLength, int b, boolean rootFirst)
    {
        this.isDecode = isDecode;
        this.classifier = classifier;
        this.configuration = configuration;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.b = b;
        this.rootFirst = rootFirst;
    }


    public ArrayList<BeamElement> call()
    {
        ArrayList<BeamElement> elements = new ArrayList<BeamElement>(dependencyRelations.size() * 2 + 3);

        State currentState = configuration.state;
        float prevScore = configuration.score;

        boolean canShift = ArcEager.canDo(Action.Shift, currentState);
        boolean canReduce = ArcEager.canDo(Action.Reduce, currentState);
        boolean canRightArc = ArcEager.canDo(Action.RightArc, currentState);
        boolean canLeftArc = ArcEager.canDo(Action.LeftArc, currentState);
        Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

        addAvailableBeamElements(elements, prevScore, canShift, canReduce, canRightArc, canLeftArc, features, classifier, isDecode, b, dependencyRelations);
        return elements;
    }
}