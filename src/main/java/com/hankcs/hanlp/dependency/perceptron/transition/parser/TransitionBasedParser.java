/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.structures.IndexMaps;

import java.util.ArrayList;

/**
 * This class is just for making connection between different types of transition-based parsers
 */
public abstract class TransitionBasedParser
{

    /**
     * Any kind of classifier that can give us scores
     */
    protected AveragedPerceptron classifier;
    protected ArrayList<Integer> dependencyRelations;
    protected int featureLength;
    protected IndexMaps maps;

    public TransitionBasedParser(AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations, int featureLength, IndexMaps maps)
    {
        this.classifier = classifier;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.maps = maps;
    }

    public String idWord(int id)
    {
        return maps.idWord[id];
    }
}
