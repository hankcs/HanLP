/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.configuration;

import com.hankcs.hanlp.dependency.perceptron.accessories.Pair;

import java.util.ArrayList;
import java.util.HashMap;

public class CompactTree
{
    public HashMap<Integer, Pair<Integer, String>> goldDependencies;
    public ArrayList<String> posTags;

    public CompactTree(HashMap<Integer, Pair<Integer, String>> goldDependencies, ArrayList<String> posTags)
    {
        this.goldDependencies = goldDependencies;
        this.posTags = posTags;
    }
}
