/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */


package com.hankcs.hanlp.dependency.perceptron.transition.configuration;

import com.hankcs.hanlp.dependency.perceptron.accessories.Edge;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.Action;
import com.hankcs.hanlp.dependency.perceptron.structures.Sentence;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.ArcEager;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * 训练实例
 */
public class Instance
{
    /**
     * dependent -> head
     */
    protected HashMap<Integer, Edge> goldDependencies;
    /**
     * head -> dependents
     */
    protected HashMap<Integer, HashSet<Integer>> reversedDependencies;
    protected Sentence sentence;

    public Instance(Sentence sentence, HashMap<Integer, Edge> goldDependencies)
    {
        this.goldDependencies = new HashMap<Integer, Edge>();
        reversedDependencies = new HashMap<Integer, HashSet<Integer>>();
        for (Map.Entry<Integer, Edge> entry : goldDependencies.entrySet())
        {
            Integer dependent = entry.getKey();
            Edge edge = entry.getValue();
            int head = edge.headIndex;
            this.goldDependencies.put(dependent, edge.clone());
            HashSet<Integer> dependents = reversedDependencies.get(head);
            if (dependents == null)
            {
                dependents = new HashSet<Integer>();
                reversedDependencies.put(head, dependents);
            }
            dependents.add(dependent);
        }
        this.sentence = sentence;
    }


    public Sentence getSentence()
    {
        return sentence;
    }

    public int head(int dependent)
    {
        if (!goldDependencies.containsKey(dependent))
            return -1;
        return goldDependencies.get(dependent).headIndex;
    }

    public String relation(int dependent)
    {
        if (!goldDependencies.containsKey(dependent))
            return "_";
        return goldDependencies.get(dependent).relationId + "";
    }

    public HashMap<Integer, Edge> getGoldDependencies()
    {
        return goldDependencies;
    }

    /**
     * Shows whether the tree to train is projective or not
     *
     * @return true if the tree is non-projective
     */
    public boolean isNonprojective()
    {
        for (int dep1 : goldDependencies.keySet())
        {
            int head1 = goldDependencies.get(dep1).headIndex;
            for (int dep2 : goldDependencies.keySet())
            {
                int head2 = goldDependencies.get(dep2).headIndex;
                if (head1 < 0 || head2 < 0)
                    continue;
                if (dep1 > head1 && head1 != head2)
                    if ((dep1 > head2 && dep1 < dep2 && head1 < head2) || (dep1 < head2 && dep1 > dep2 && head1 < dep2))
                        return true;
                if (dep1 < head1 && head1 != head2)
                    if ((head1 > head2 && head1 < dep2 && dep1 < head2) || (head1 < head2 && head1 > dep2 && dep1 < dep2))
                        return true;
            }
        }
        return false;
    }

    public boolean isPartial(boolean rootFirst)
    {
        for (int i = 0; i < sentence.size(); i++)
        {
            if (rootFirst || i < sentence.size() - 1)
            {
                if (!goldDependencies.containsKey(i + 1))
                    return true;
            }
        }
        return false;
    }

    public HashMap<Integer, HashSet<Integer>> getReversedDependencies()
    {
        return reversedDependencies;
    }

    /**
     * For the cost of an action given the gold dependencies
     * For more information see:
     * Yoav Goldberg and Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic Oracles."
     * TACL 1 (2013): 403-414.
     *
     * @param action
     * @param dependency
     * @param state
     * @return oracle cost of the action
     * @throws Exception
     */
    public int actionCost(Action action, int dependency, State state)
    {
        if (!ArcEager.canDo(action, state))
            return Integer.MAX_VALUE;
        int cost = 0;

        // added by me to take care of labels
        if (action == Action.LeftArc)
        { // left arc
            int bufferHead = state.bufferHead();
            int stackHead = state.stackTop();

            if (goldDependencies.containsKey(stackHead) && goldDependencies.get(stackHead).headIndex == bufferHead
                    && goldDependencies.get(stackHead).relationId != (dependency))
                cost += 1;
        }
        else if (action == Action.RightArc)
        { //right arc
            int bufferHead = state.bufferHead();
            int stackHead = state.stackTop();
            if (goldDependencies.containsKey(bufferHead) && goldDependencies.get(bufferHead).headIndex == stackHead
                    && goldDependencies.get(bufferHead).relationId != (dependency))
                cost += 1;
        }

        if (action == Action.Shift)
        { //shift
            int bufferHead = state.bufferHead();
            for (int stackItem : state.getStack())
            {
                if (goldDependencies.containsKey(stackItem) && goldDependencies.get(stackItem).headIndex == (bufferHead))
                    cost += 1;
                if (goldDependencies.containsKey(bufferHead) && goldDependencies.get(bufferHead).headIndex == (stackItem))
                    cost += 1;
            }

        }
        else if (action == Action.Reduce)
        { //reduce
            int stackHead = state.stackTop();
            if (!state.bufferEmpty())
                for (int bufferItem = state.bufferHead(); bufferItem <= state.maxSentenceSize; bufferItem++)
                {
                    if (goldDependencies.containsKey(bufferItem) && goldDependencies.get(bufferItem).headIndex == (stackHead))
                        cost += 1;
                }
        }
        else if (action == Action.LeftArc && cost == 0)
        { //left arc
            int stackHead = state.stackTop();
            if (!state.bufferEmpty())
                for (int bufferItem = state.bufferHead(); bufferItem <= state.maxSentenceSize; bufferItem++)
                {
                    if (goldDependencies.containsKey(bufferItem) && goldDependencies.get(bufferItem).headIndex == (stackHead))
                        cost += 1;
                    if (goldDependencies.containsKey(stackHead) && goldDependencies.get(stackHead).headIndex == (bufferItem))
                        if (bufferItem != state.bufferHead())
                            cost += 1;
                }
        }
        else if (action == Action.RightArc && cost == 0)
        { //right arc
            int stackHead = state.stackTop();
            int bufferHead = state.bufferHead();
            for (int stackItem : state.getStack())
            {
                if (goldDependencies.containsKey(bufferHead) && goldDependencies.get(bufferHead).headIndex == (stackItem))
                    if (stackItem != stackHead)
                        cost += 1;

                if (goldDependencies.containsKey(stackItem) && goldDependencies.get(stackItem).headIndex == (bufferHead))
                    cost += 1;
            }
            if (!state.bufferEmpty())
                for (int bufferItem = state.bufferHead(); bufferItem <= state.maxSentenceSize; bufferItem++)
                {
                    if (goldDependencies.containsKey(bufferHead) && goldDependencies.get(bufferHead).headIndex == (bufferItem))
                        cost += 1;
                }
        }
        return cost;
    }
}
