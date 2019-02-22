/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */


package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.accessories.Pair;
import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Instance;
import com.hankcs.hanlp.dependency.perceptron.transition.features.FeatureExtractor;
import com.hankcs.hanlp.dependency.perceptron.structures.Sentence;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.BeamElement;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.State;

import java.util.ArrayList;
import java.util.Collection;
import java.util.TreeSet;
import java.util.concurrent.Callable;

public class ParseThread implements Callable<Pair<Configuration, Integer>>
{
    AveragedPerceptron classifier;

    ArrayList<Integer> dependencyRelations;

    int featureLength;

    Sentence sentence;
    boolean rootFirst;
    int beamWidth;
    Instance instance;
    boolean partial;

    int id;

    public ParseThread(int id, AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations, int featureLength,
                       Sentence sentence,
                       boolean rootFirst, int beamWidth, Instance instance, boolean partial)
    {
        this.id = id;
        this.classifier = classifier;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.sentence = sentence;
        this.rootFirst = rootFirst;
        this.beamWidth = beamWidth;
        this.instance = instance;
        this.partial = partial;
    }

    @Override
    public Pair<Configuration, Integer> call() throws Exception
    {
        if (!partial)
            return parse();
        else return new Pair<Configuration, Integer>(parsePartial(), id);
    }

    Pair<Configuration, Integer> parse() throws Exception
    {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);

        ArrayList<Configuration> beam = new ArrayList<Configuration>(beamWidth);
        beam.add(initialConfiguration);

        while (!ArcEager.isTerminal(beam))
        {
            if (beamWidth != 1)
            {
                TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();
                sortBeam(beam, beamPreserver, false, null, beamWidth, rootFirst, featureLength, classifier, dependencyRelations);

                ArrayList<Configuration> repBeam = new ArrayList<Configuration>(beamWidth);
                for (BeamElement beamElement : beamPreserver.descendingSet())
                {
                    if (repBeam.size() >= beamWidth)
                        break;
                    int b = beamElement.index;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    float score = beamElement.score;

                    Configuration newConfig = beam.get(b).clone();

                    if (action == 0)
                    {
                        ArcEager.shift(newConfig.state);
                        newConfig.addAction(0);
                    }
                    else if (action == 1)
                    {
                        ArcEager.reduce(newConfig.state);
                        newConfig.addAction(1);
                    }
                    else if (action == 2)
                    {
                        ArcEager.rightArc(newConfig.state, label);
                        newConfig.addAction(3 + label);
                    }
                    else if (action == 3)
                    {
                        ArcEager.leftArc(newConfig.state, label);
                        newConfig.addAction(3 + dependencyRelations.size() + label);
                    }
                    else if (action == 4)
                    {
                        ArcEager.unShift(newConfig.state);
                        newConfig.addAction(2);
                    }
                    newConfig.setScore(score);
                    repBeam.add(newConfig);
                }
                beam = repBeam;
            }
            else
            {
                Configuration configuration = beam.get(0);
                State currentState = configuration.state;
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                float bestScore = Float.NEGATIVE_INFINITY;
                int bestAction = -1;

                boolean canShift = ArcEager.canDo(Action.Shift, currentState);
                boolean canReduce = ArcEager.canDo(Action.Reduce, currentState);
                boolean canRightArc = ArcEager.canDo(Action.RightArc, currentState);
                boolean canLeftArc = ArcEager.canDo(Action.LeftArc, currentState);

                if (!canShift
                        && !canReduce
                        && !canRightArc
                        && !canLeftArc)
                {

                    if (!currentState.stackEmpty())
                    {
                        ArcEager.unShift(currentState);
                        configuration.addAction(2);
                    }
                    else if (!currentState.bufferEmpty() && currentState.stackEmpty())
                    {
                        ArcEager.shift(currentState);
                        configuration.addAction(0);
                    }
                }

                if (canShift)
                {
                    float score = classifier.shiftScore(features, true);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestAction = 0;
                    }
                }
                if (canReduce)
                {
                    float score = classifier.reduceScore(features, true);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestAction = 1;
                    }
                }
                if (canRightArc)
                {
                    float[] rightArcScores = classifier.rightArcScores(features, true);
                    for (int dependency : dependencyRelations)
                    {
                        float score = rightArcScores[dependency];
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestAction = 3 + dependency;
                        }
                    }
                }
                if (ArcEager.canDo(Action.LeftArc, currentState))
                {
                    float[] leftArcScores = classifier.leftArcScores(features, true);
                    for (int dependency : dependencyRelations)
                    {
                        float score = leftArcScores[dependency];
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestAction = 3 + dependencyRelations.size() + dependency;
                        }
                    }
                }

                if (bestAction != -1)
                {
                    if (bestAction == 0)
                    {
                        ArcEager.shift(configuration.state);
                    }
                    else if (bestAction == (1))
                    {
                        ArcEager.reduce(configuration.state);
                    }
                    else
                    {

                        if (bestAction >= 3 + dependencyRelations.size())
                        {
                            int label = bestAction - (3 + dependencyRelations.size());
                            ArcEager.leftArc(configuration.state, label);
                        }
                        else
                        {
                            int label = bestAction - 3;
                            ArcEager.rightArc(configuration.state, label);
                        }
                    }
                    configuration.addScore(bestScore);
                    configuration.addAction(bestAction);
                }
                if (beam.size() == 0)
                {
                    System.out.println("WHY BEAM SIZE ZERO?");
                }
            }
        }

        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam)
        {
            if (configuration.getScore(true) > bestScore)
            {
                bestScore = configuration.getScore(true);
                bestConfiguration = configuration;
            }
        }
        return new Pair<Configuration, Integer>(bestConfiguration, id);
    }

    public static void sortBeam(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, Boolean isNonProjective, Instance instance, int beamWidth, boolean rootFirst, int featureLength, AveragedPerceptron classifier, Collection<Integer> dependencyRelations)
    {
        for (int b = 0; b < beam.size(); b++)
        {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Action.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Action.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Action.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Action.LeftArc, currentState);
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
            if (!canShift
                    && !canReduce
                    && !canRightArc
                    && !canLeftArc)
            {
                beamPreserver.add(new BeamElement(prevScore, b, 4, -1));

                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canShift)
            {
                float score = classifier.shiftScore(features, true);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canReduce)
            {
                float score = classifier.reduceScore(features, true);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canRightArc)
            {
                float[] rightArcScores = classifier.rightArcScores(features, true);
                for (int dependency : dependencyRelations)
                {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }

            if (canLeftArc)
            {
                float[] leftArcScores = classifier.leftArcScores(features, true);
                for (int dependency : dependencyRelations)
                {
                    float score = leftArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }
        }
    }

    public Configuration parsePartial() throws Exception
    {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        boolean isNonProjective = false;
        if (instance.isNonprojective())
        {
            isNonProjective = true;
        }

        ArrayList<Configuration> beam = new ArrayList<Configuration>(beamWidth);
        beam.add(initialConfiguration);

        while (!ArcEager.isTerminal(beam))
        {
            TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();

            sortBeam(beam, beamPreserver, isNonProjective, instance, beamWidth, rootFirst, featureLength, classifier, dependencyRelations);

            ArrayList<Configuration> repBeam = new ArrayList<Configuration>(beamWidth);
            for (BeamElement beamElement : beamPreserver.descendingSet())
            {
                if (repBeam.size() >= beamWidth)
                    break;
                int b = beamElement.index;
                int action = beamElement.action;
                int label = beamElement.label;
                float score = beamElement.score;

                Configuration newConfig = beam.get(b).clone();

                if (action == 0)
                {
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                }
                else if (action == 1)
                {
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                }
                else if (action == 2)
                {
                    ArcEager.rightArc(newConfig.state, label);
                    newConfig.addAction(3 + label);
                }
                else if (action == 3)
                {
                    ArcEager.leftArc(newConfig.state, label);
                    newConfig.addAction(3 + dependencyRelations.size() + label);
                }
                else if (action == 4)
                {
                    ArcEager.unShift(newConfig.state);
                    newConfig.addAction(2);
                }
                newConfig.setScore(score);
                repBeam.add(newConfig);
            }
            beam = repBeam;
        }

        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam)
        {
            if (configuration.getScore(true) > bestScore)
            {
                bestScore = configuration.getScore(true);
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }
}