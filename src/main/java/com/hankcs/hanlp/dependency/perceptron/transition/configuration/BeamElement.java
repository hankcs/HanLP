/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.configuration;

public class BeamElement implements Comparable<BeamElement>
{
    public float score;
    public int index;
    public int action;
    public int label;

    public BeamElement(float score, int index, int action, int label)
    {
        this.score = score;
        this.index = index;
        this.action = action;
        this.label = label;
    }

    @Override
    public int compareTo(BeamElement beamElement)
    {
        float diff = score - beamElement.score;
        if (diff > 0)
            return 2;
        if (diff < 0)
            return -2;
        if (index != beamElement.index)
            return beamElement.index - index;
        return beamElement.action - action;
    }

    @Override
    public boolean equals(Object o)
    {
        return false;
    }
}
