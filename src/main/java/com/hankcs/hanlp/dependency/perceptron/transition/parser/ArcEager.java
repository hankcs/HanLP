/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.structures.IndexMaps;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.State;

import java.util.ArrayList;

public class ArcEager extends TransitionBasedParser
{
    private ArcEager(AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations, int featureLength, IndexMaps maps)
    {
        super(classifier, dependencyRelations, featureLength, maps);
    }

    public static void shift(State state)
    {
        state.push(state.bufferHead());
        state.incrementBufferHead();

        // changing the constraint
        if (state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public static void unShift(State state) 
    {
        if (!state.stackEmpty())
            state.setBufferHead(state.pop());
        // to make sure
        state.setEmptyFlag(true);
        state.setMaxSentenceSize(state.bufferHead());
    }

    public static void reduce(State state) 
    {
        state.pop();
        if (state.stackEmpty() && state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public static void leftArc(State state, int dependency) 
    {
        state.addArc(state.pop(), state.bufferHead(), dependency);
    }

    public static void rightArc(State state, int dependency) 
    {
        state.addArc(state.bufferHead(), state.stackTop(), dependency);
        state.push(state.bufferHead());
        state.incrementBufferHead();
        if (!state.isEmptyFlag() && state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public static boolean canDo(Action action, State state)
    {
        if (action == Action.Shift)
        { //shift
            return !(!state.bufferEmpty() && state.bufferHead() == state.rootIndex && !state.stackEmpty()) && !state.bufferEmpty() && !state.isEmptyFlag();
        }
        else if (action == Action.RightArc)
        { //right arc
            if (state.stackEmpty())
                return false;
            return !(!state.bufferEmpty() && state.bufferHead() == state.rootIndex) && !state.bufferEmpty() && !state.stackEmpty();

        }
        else if (action == Action.LeftArc)
        { //left arc
            if (state.stackEmpty() || state.bufferEmpty())
                return false;

            if (!state.stackEmpty() && state.stackTop() == state.rootIndex)
                return false;

            return state.stackTop() != state.rootIndex && !state.hasHead(state.stackTop()) && !state.stackEmpty();
        }
        else if (action == Action.Reduce)
        { //reduce
            return !state.stackEmpty() && state.hasHead(state.stackTop()) || !state.stackEmpty() && state.stackSize() == 1 && state.bufferSize() == 0 && state.stackTop() == state.rootIndex;
        }
        else if (action == Action.Unshift)
        { //unshift
            return !state.stackEmpty() && !state.hasHead(state.stackTop()) && state.isEmptyFlag();
        }
        return false;
    }

    /**
     * Shows true if all of the configurations in the beam are in the terminal state
     *
     * @param beam the current beam
     * @return true if all of the configurations in the beam are in the terminal state
     */
    public static boolean isTerminal(ArrayList<Configuration> beam)
    {
        for (Configuration configuration : beam)
            if (!configuration.state.isTerminalState())
                return false;
        return true;
    }


    public static void commitAction(int action, int label, float score, ArrayList<Integer> dependencyRelations, Configuration newConfig) 
    {
        if (action == 0)
        {
            shift(newConfig.state);
            newConfig.addAction(0);
        }
        else if (action == 1)
        {
            reduce(newConfig.state);
            newConfig.addAction(1);
        }
        else if (action == 2)
        {
            rightArc(newConfig.state, label);
            newConfig.addAction(3 + label);
        }
        else if (action == 3)
        {
            leftArc(newConfig.state, label);
            newConfig.addAction(3 + dependencyRelations.size() + label);
        }
        else if (action == 4)
        {
            unShift(newConfig.state);
            newConfig.addAction(2);
        }
        newConfig.setScore(score);
    }
}
