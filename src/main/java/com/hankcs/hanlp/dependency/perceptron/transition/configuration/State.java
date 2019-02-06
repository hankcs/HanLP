/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.configuration;

import com.hankcs.hanlp.dependency.perceptron.accessories.Edge;

import java.util.ArrayDeque;

/**
 * 由buffer、stack和arc组成的状态
 */
public class State implements Cloneable
{
    public int rootIndex;
    public int maxSentenceSize;

    /**
     * This is the additional information for the case of parsing with tree constraint
     * For more information see:
     * Joakim Nivre and Daniel FernÃ¡ndez-GonzÃ¡lez. "Arc-Eager Parsing with the Tree Constraint."
     * Computational Linguistics(2014).
     */
    protected boolean emptyFlag;

    /**
     * Keeps dependent->head information
     */
    protected Edge[] arcs;
    protected int[] leftMostArcs;
    protected int[] rightMostArcs;
    /**
     * left modifiers
     */
    protected int[] leftValency;
    protected int[] rightValency;
    protected long[] rightDepLabels;
    protected long[] leftDepLabels;
    protected ArrayDeque<Integer> stack;
    int bufferHead;

    public State(int size)
    {
        emptyFlag = false;
        stack = new ArrayDeque<Integer>();
        arcs = new Edge[size + 1];

        leftMostArcs = new int[size + 1];
        rightMostArcs = new int[size + 1];
        leftValency = new int[size + 1];
        rightValency = new int[size + 1];
        rightDepLabels = new long[size + 1];
        leftDepLabels = new long[size + 1];

        rootIndex = 0;
        bufferHead = 1;
        maxSentenceSize = 0;
    }

    /**
     * @param sentenceSize 句子长度（不包含ROOT）
     * @param rootFirst    是否将ROOT作为index=0的词语，否则作为最后一个词语
     */
    public State(int sentenceSize, boolean rootFirst)
    {
        this(sentenceSize);
        if (rootFirst)
        {
            stack.push(0);
            rootIndex = 0;
            maxSentenceSize = sentenceSize;
        }
        else
        {
            rootIndex = sentenceSize;
            maxSentenceSize = sentenceSize;
        }
    }

    public ArrayDeque<Integer> getStack()
    {
        return stack;
    }

    public int pop() 
    {
        return stack.pop();
    }

    public void push(int index)
    {
        stack.push(index);
    }

    public void addArc(int dependent, int head, int dependency)
    {
        arcs[dependent] = new Edge(head, dependency);
        long value = 1L << (dependency);

        assert dependency < 64;

        if (dependent > head)
        { //right dep
            if (rightMostArcs[head] == 0 || dependent > rightMostArcs[head])
                rightMostArcs[head] = dependent;
            rightValency[head] += 1;
            rightDepLabels[head] = rightDepLabels[head] | value;

        }
        else
        { //left dependency
            if (leftMostArcs[head] == 0 || dependent < leftMostArcs[head])
                leftMostArcs[head] = dependent;
            leftDepLabels[head] = leftDepLabels[head] | value;
            leftValency[head] += 1;
        }
    }

    public long rightDependentLabels(int position)
    {
        return rightDepLabels[position];
    }

    public long leftDependentLabels(int position)
    {
        return leftDepLabels[position];
    }

    public boolean isEmptyFlag()
    {
        return emptyFlag;
    }

    public void setEmptyFlag(boolean emptyFlag)
    {
        this.emptyFlag = emptyFlag;
    }

    public int bufferHead()
    {
        return bufferHead;
    }

    /**
     * View top element of stack
     * @return
     */
    public int stackTop()
    {
        if (stack.size() > 0)
            return stack.peek();
        return -1;
    }

    public int getBufferItem(int position)
    {
        return bufferHead + position;
    }

    public boolean isTerminalState()
    {
        if (stackEmpty())
        {
            if (bufferEmpty() || bufferHead == rootIndex)
            {
                return true;
            }
        }
        return false;
    }

    public boolean hasHead(int dependent)
    {
        return arcs[dependent] != null;
    }

    public boolean bufferEmpty()
    {
        return bufferHead == -1;
    }

    public boolean stackEmpty()
    {
        return stack.size() == 0;
    }

    public int bufferSize()
    {
        if (bufferHead < 0)
            return 0;
        return (maxSentenceSize - bufferHead + 1);
    }

    public int stackSize()
    {
        return stack.size();
    }

    public int rightMostModifier(int index)
    {
        return (rightMostArcs[index] == 0 ? -1 : rightMostArcs[index]);
    }

    public int leftMostModifier(int index)
    {
        return (leftMostArcs[index] == 0 ? -1 : leftMostArcs[index]);
    }

    /**
     * @param head
     * @return the current index of dependents
     */
    public int valence(int head)
    {
        return rightValency(head) + leftValency(head);
    }

    /**
     * @param head
     * @return the current index of right modifiers
     */
    public int rightValency(int head)
    {
        return rightValency[head];
    }

    /**
     * @param head
     * @return the current index of left modifiers
     */
    public int leftValency(int head)
    {
        return leftValency[head];
    }

    public int getHead(int index)
    {
        if (arcs[index] != null)
            return arcs[index].headIndex;
        return -1;
    }

    public int getDependent(int index)
    {
        if (arcs[index] != null)
            return arcs[index].relationId;
        return -1;
    }

    public void setMaxSentenceSize(int maxSentenceSize)
    {
        this.maxSentenceSize = maxSentenceSize;
    }

    public void incrementBufferHead()
    {
        if (bufferHead == maxSentenceSize)
            bufferHead = -1;
        else
            bufferHead++;
    }

    public void setBufferHead(int bufferHead)
    {
        this.bufferHead = bufferHead;
    }

    @Override
    public State clone()
    {
        State state = new State(arcs.length - 1);
        state.stack = new ArrayDeque<Integer>(stack);

        for (int dependent = 0; dependent < arcs.length; dependent++)
        {
            if (arcs[dependent] != null)
            {
                Edge head = arcs[dependent];
                state.arcs[dependent] = head;
                int h = head.headIndex;

                if (rightMostArcs[h] != 0)
                {
                    state.rightMostArcs[h] = rightMostArcs[h];
                    state.rightValency[h] = rightValency[h];
                    state.rightDepLabels[h] = rightDepLabels[h];
                }

                if (leftMostArcs[h] != 0)
                {
                    state.leftMostArcs[h] = leftMostArcs[h];
                    state.leftValency[h] = leftValency[h];
                    state.leftDepLabels[h] = leftDepLabels[h];
                }
            }
        }
        state.rootIndex = rootIndex;
        state.bufferHead = bufferHead;
        state.maxSentenceSize = maxSentenceSize;
        state.emptyFlag = emptyFlag;
        return state;
    }
}
