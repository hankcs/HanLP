/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.learning;

import com.hankcs.hanlp.dependency.perceptron.structures.ParserModel;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.Action;
import com.hankcs.hanlp.dependency.perceptron.structures.CompactArray;

import java.util.HashMap;

public class AveragedPerceptron
{
    /**
     * This class tries to implement averaged Perceptron algorithm
     * Collins, Michael. "Discriminative training methods for hidden Markov models: Theory and experiments with Perceptron algorithms."
     * In Proceedings of the ACL-02 conference on Empirical methods in natural language processing-Volume 10, pp. 1-8.
     * Association for Computational Linguistics, 2002.
     * <p/>
     * The averaging update is also optimized by using the trick introduced in Hal Daume's dissertation.
     * For more information see the second chapter of his thesis:
     * Harold Charles Daume' III. "Practical Structured YaraParser.Learning Techniques for Natural Language Processing", PhD thesis, ISI USC, 2006.
     * http://www.umiacs.umd.edu/~hal/docs/daume06thesis.pdf
     */
    /**
     * For the weights for all features
     */
    public HashMap<Object, Float>[] shiftFeatureWeights;
    public HashMap<Object, Float>[] reduceFeatureWeights;
    public HashMap<Object, CompactArray>[] leftArcFeatureWeights;
    public HashMap<Object, CompactArray>[] rightArcFeatureWeights;

    public int iteration;
    public int dependencySize;
    /**
     * This is the main part of the extension to the original perceptron algorithm which the averaging over all the history
     */
    public HashMap<Object, Float>[] shiftFeatureAveragedWeights;
    public HashMap<Object, Float>[] reduceFeatureAveragedWeights;
    public HashMap<Object, CompactArray>[] leftArcFeatureAveragedWeights;
    public HashMap<Object, CompactArray>[] rightArcFeatureAveragedWeights;


    public AveragedPerceptron(int featSize, int dependencySize)
    {
        shiftFeatureWeights = new HashMap[featSize];
        reduceFeatureWeights = new HashMap[featSize];
        leftArcFeatureWeights = new HashMap[featSize];
        rightArcFeatureWeights = new HashMap[featSize];

        shiftFeatureAveragedWeights = new HashMap[featSize];
        reduceFeatureAveragedWeights = new HashMap[featSize];
        leftArcFeatureAveragedWeights = new HashMap[featSize];
        rightArcFeatureAveragedWeights = new HashMap[featSize];
        for (int i = 0; i < featSize; i++)
        {
            shiftFeatureWeights[i] = new HashMap<Object, Float>();
            reduceFeatureWeights[i] = new HashMap<Object, Float>();
            leftArcFeatureWeights[i] = new HashMap<Object, CompactArray>();
            rightArcFeatureWeights[i] = new HashMap<Object, CompactArray>();


            shiftFeatureAveragedWeights[i] = new HashMap<Object, Float>();
            reduceFeatureAveragedWeights[i] = new HashMap<Object, Float>();
            leftArcFeatureAveragedWeights[i] = new HashMap<Object, CompactArray>();
            rightArcFeatureAveragedWeights[i] = new HashMap<Object, CompactArray>();
        }

        iteration = 1;
        this.dependencySize = dependencySize;
    }

    private AveragedPerceptron(HashMap<Object, Float>[] shiftFeatureAveragedWeights, HashMap<Object, Float>[] reduceFeatureAveragedWeights,
                               HashMap<Object, CompactArray>[] leftArcFeatureAveragedWeights, HashMap<Object, CompactArray>[] rightArcFeatureAveragedWeights,
                               int dependencySize)
    {
        this.shiftFeatureAveragedWeights = shiftFeatureAveragedWeights;
        this.reduceFeatureAveragedWeights = reduceFeatureAveragedWeights;
        this.leftArcFeatureAveragedWeights = leftArcFeatureAveragedWeights;
        this.rightArcFeatureAveragedWeights = rightArcFeatureAveragedWeights;
        this.dependencySize = dependencySize;
    }

    public AveragedPerceptron(ParserModel parserModel)
    {
        this(parserModel.shiftFeatureAveragedWeights, parserModel.reduceFeatureAveragedWeights, parserModel.leftArcFeatureAveragedWeights, parserModel.rightArcFeatureAveragedWeights, parserModel.dependencySize);
    }

    public float changeWeight(Action actionType, int slotNum, Object featureName, int labelIndex, float change)
    {
        if (featureName == null)
            return 0;
        if (actionType == Action.Shift)
        {
            if (!shiftFeatureWeights[slotNum].containsKey(featureName))
                shiftFeatureWeights[slotNum].put(featureName, change);
            else
                shiftFeatureWeights[slotNum].put(featureName, shiftFeatureWeights[slotNum].get(featureName) + change);

            if (!shiftFeatureAveragedWeights[slotNum].containsKey(featureName))
                shiftFeatureAveragedWeights[slotNum].put(featureName, iteration * change);
            else
                shiftFeatureAveragedWeights[slotNum].put(featureName, shiftFeatureAveragedWeights[slotNum].get(featureName) + iteration * change);
        }
        else if (actionType == Action.Reduce)
        {
            if (!reduceFeatureWeights[slotNum].containsKey(featureName))
                reduceFeatureWeights[slotNum].put(featureName, change);
            else
                reduceFeatureWeights[slotNum].put(featureName, reduceFeatureWeights[slotNum].get(featureName) + change);

            if (!reduceFeatureAveragedWeights[slotNum].containsKey(featureName))
                reduceFeatureAveragedWeights[slotNum].put(featureName, iteration * change);
            else
                reduceFeatureAveragedWeights[slotNum].put(featureName, reduceFeatureAveragedWeights[slotNum].get(featureName) + iteration * change);
        }
        else if (actionType == Action.RightArc)
        {
            changeFeatureWeight(rightArcFeatureWeights[slotNum], rightArcFeatureAveragedWeights[slotNum], featureName, labelIndex, change, dependencySize);
        }
        else if (actionType == Action.LeftArc)
        {
            changeFeatureWeight(leftArcFeatureWeights[slotNum], leftArcFeatureAveragedWeights[slotNum], featureName, labelIndex, change, dependencySize);
        }

        return change;
    }

    public void changeFeatureWeight(HashMap<Object, CompactArray> map, HashMap<Object, CompactArray> aMap, Object featureName, int labelIndex, float change, int size)
    {
        CompactArray values = map.get(featureName);
        CompactArray aValues;
        if (values != null)
        {
            values.set(labelIndex, change);
            aValues = aMap.get(featureName);
            aValues.set(labelIndex, iteration * change);
        }
        else
        {
            float[] val = new float[]{change};
            values = new CompactArray(labelIndex, val);
            map.put(featureName, values);

            float[] aVal = new float[]{iteration * change};
            aValues = new CompactArray(labelIndex, aVal);
            aMap.put(featureName, aValues);
        }
    }


    /**
     * Adds to the iterations
     */
    public void incrementIteration()
    {
        iteration++;
    }

    public float shiftScore(final Object[] features, boolean decode)
    {
        float score = 0.0f;

        HashMap<Object, Float>[] map = decode ? shiftFeatureAveragedWeights : shiftFeatureWeights;

        for (int i = 0; i < features.length; i++)
        {
            if (features[i] == null || (i >= 26 && i < 32)) // [26, 32) is distance feature
                continue;
            Float weight = map[i].get(features[i]);
            if (weight != null)
            {
                score += weight;
            }
        }

        return score;
    }

    public float reduceScore(final Object[] features, boolean decode)
    {
        float score = 0.0f;

        HashMap<Object, Float>[] map = decode ? reduceFeatureAveragedWeights : reduceFeatureWeights;

        for (int i = 0; i < features.length; i++)
        {
            if (features[i] == null || (i >= 26 && i < 32))
                continue;
            Float values = map[i].get(features[i]);
            if (values != null)
            {
                score += values;
            }
        }

        return score;
    }

    public float[] leftArcScores(final Object[] features, boolean decode)
    {
        float scores[] = new float[dependencySize];

        HashMap<Object, CompactArray>[] map = decode ? leftArcFeatureAveragedWeights : leftArcFeatureWeights;

        for (int i = 0; i < features.length; i++)
        {
            if (features[i] == null)
                continue;
            CompactArray values = map[i].get(features[i]);
            if (values != null)
            {
                int offset = values.getOffset();
                float[] weightVector = values.getArray();

                for (int d = offset; d < offset + weightVector.length; d++)
                {
                    scores[d] += weightVector[d - offset];
                }
            }
        }

        return scores;
    }

    public float[] rightArcScores(final Object[] features, boolean decode)
    {
        float scores[] = new float[dependencySize];

        HashMap<Object, CompactArray>[] map = decode ? rightArcFeatureAveragedWeights : rightArcFeatureWeights;

        for (int i = 0; i < features.length; i++)
        {
            if (features[i] == null)
                continue;
            CompactArray values = map[i].get(features[i]);
            if (values != null)
            {
                int offset = values.getOffset();
                float[] weightVector = values.getArray();

                for (int d = offset; d < offset + weightVector.length; d++)
                {
                    scores[d] += weightVector[d - offset];
                }
            }
        }

        return scores;
    }

    public int featureSize()
    {
        return shiftFeatureAveragedWeights.length;
    }

    public int raSize()
    {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++)
        {
            for (Object feat : rightArcFeatureAveragedWeights[i].keySet())
            {
                size += rightArcFeatureAveragedWeights[i].get(feat).length();
            }
        }
        return size;
    }

    public int effectiveRaSize()
    {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++)
        {
            for (Object feat : rightArcFeatureAveragedWeights[i].keySet())
            {
                for (float f : rightArcFeatureAveragedWeights[i].get(feat).getArray())
                    if (f != 0f)
                        size++;
            }
        }
        return size;
    }


    public int laSize()
    {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++)
        {
            for (Object feat : leftArcFeatureAveragedWeights[i].keySet())
            {
                size += leftArcFeatureAveragedWeights[i].get(feat).length();
            }
        }
        return size;
    }

    public int effectiveLaSize()
    {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++)
        {
            for (Object feat : leftArcFeatureAveragedWeights[i].keySet())
            {
                for (float f : leftArcFeatureAveragedWeights[i].get(feat).getArray())
                    if (f != 0f)
                        size++;
            }
        }
        return size;
    }
}
