/**
 * MDAG is a Java library capable of constructing character-sequence-storing,
 * directed acyclic graphs of minimal size. 
 *
 *  Copyright (C) 2012 Kevin Lawson <Klawson88@gmail.com>
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.hankcs.hanlp.collection.MDAG;


import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;

import java.io.DataOutputStream;

/**
 * The class capable of representing a MDAG node, its _transition set, and one of its incoming transitions;
 * objects of this class are used to represent a MDAG after its been simplified in order to save space.
 *
 * @author Kevin
 */
public class SimpleMDAGNode implements ICacheAble
{
    //The character labeling an incoming _transition to this node
    private char letter;

    //The boolean denoting the accept state status of this node
    private boolean isAcceptNode;

    //The int denoting the size of this node's outgoing _transition set
    private int transitionSetSize;

    //The int denoting the index (in the array which contains this node) at which this node's _transition set begins
    private int transitionSetBeginIndex;


    /**
     * Constructs a SimpleMDAGNode.
     *
     * @param letter            a char representing the _transition label leading to this SimpleMDAGNode
     * @param isAcceptNode      a boolean representing the accept state status of this SimpleMDAGNode
     * @param transitionSetSize an int denoting the size of this _transition set
     */
    public SimpleMDAGNode(char letter, boolean isAcceptNode, int transitionSetSize)
    {
        this.letter = letter;
        this.isAcceptNode = isAcceptNode;
        this.transitionSetSize = transitionSetSize;
        this.transitionSetBeginIndex = 0;           //will be changed for all objects of this type, necessary for dummy root node creation
    }

    public SimpleMDAGNode()
    {

    }


    /**
     * Retrieves the character representing the _transition laben leading up to this node.
     *
     * @return the char representing the _transition label leading up to this node
     */
    public char getLetter()
    {
        return letter;
    }


    /**
     * Retrieves the accept state status of this node.
     *
     * @return true if this node is an accept state, false otherwise
     */
    public boolean isAcceptNode()
    {
        return isAcceptNode;
    }


    /**
     * Retrieves the index in this node's containing array that its _transition set begins at.
     *
     * @return an int of the index in this node's containing array at which its _transition set begins
     */
    public int getTransitionSetBeginIndex()
    {
        return transitionSetBeginIndex;
    }


    /**
     * Retrieves the size of this node's outgoing _transition set.
     *
     * @return an int denoting the size of this node's outgoing _transition set
     */
    public int getOutgoingTransitionSetSize()
    {
        return transitionSetSize;
    }


    /**
     * Records the index in this node's containing array that its _transition set begins at.
     *
     * @param transitionSetBeginIndex an int denoting the index in this node's containing array that is _transition set beings at
     */
    public void setTransitionSetBeginIndex(int transitionSetBeginIndex)
    {
        this.transitionSetBeginIndex = transitionSetBeginIndex;
    }


    /**
     * Follows an outgoing _transition from this node.
     *
     * @param mdagDataArray the array of SimpleMDAGNodes containing this node
     * @param letter        the char representation of the desired _transition's label
     * @return the SimpleMDAGNode that is the target of the _transition labeled with {@code letter},
     * or null if there is no such labeled _transition from this node
     */
    public SimpleMDAGNode transition(SimpleMDAGNode[] mdagDataArray, char letter)
    {
        SimpleMDAGNode targetNode = null;
        int offset = binarySearch(mdagDataArray, letter);
        if (offset >= 0)
        {
            targetNode = mdagDataArray[offset];
        }
        /////

        return targetNode;
    }

    private SimpleMDAGNode transitionBruteForce(SimpleMDAGNode[] mdagDataArray, char letter)
    {
        int onePastTransitionSetEndIndex = transitionSetBeginIndex + transitionSetSize;
        SimpleMDAGNode targetNode = null;

        //Loop through the SimpleMDAGNodes in this node's _transition set, searching for
        //the one with a letter equal to that which labels the desired _transition
        for(int i = transitionSetBeginIndex; i < onePastTransitionSetEndIndex; i++)
        {
            if(mdagDataArray[i].getLetter() == letter)
            {
                targetNode = mdagDataArray[i];
                break;
            }
        }
        /////

        return targetNode;
    }

    /**
     * 二分搜索
     * @param mdagDataArray
     * @param node
     * @return
     */
    private int binarySearch(SimpleMDAGNode[] mdagDataArray, char node)
    {
        if (transitionSetSize < 1)
        {
            return -1;
        }
        int high = transitionSetBeginIndex + transitionSetSize - 1;
        int low = transitionSetBeginIndex;
        while (low <= high)
        {
            int mid = ((low + high) >>> 1);
            int cmp = mdagDataArray[mid].getLetter() - node;

            if (cmp < 0)
                low = mid + 1;
            else if (cmp > 0)
                high = mid - 1;
            else
                return mid;
        }
        return -1;
    }


    /**
     * Follows a _transition path starting from this node.
     *
     * @param mdagDataArray the array of SimpleMDAGNodes containing this node
     * @param str           a String corresponding a _transition path in the MDAG
     * @return the SimpleMDAGNode at the end of the _transition path corresponding to
     * {@code str}, or null if such a _transition path is not present in the MDAG
     */
    public SimpleMDAGNode transition(SimpleMDAGNode[] mdagDataArray, String str)
    {
        SimpleMDAGNode currentNode = this;
        int numberOfChars = str.length();

        //Iteratively _transition through the MDAG using the chars in str
        for(int i = 0; i < numberOfChars; i++)
        {
            currentNode = currentNode.transition(mdagDataArray, str.charAt(i));
            if(currentNode == null) break;
        }
        /////

        return currentNode;
    }

    public SimpleMDAGNode transition(SimpleMDAGNode[] mdagDataArray, char[] str)
    {
        SimpleMDAGNode currentNode = this;
        int numberOfChars = str.length;

        //Iteratively _transition through the MDAG using the chars in str
        for (int i = 0; i < numberOfChars; i++)
        {
            currentNode = currentNode.transition(mdagDataArray, str[i]);
            if (currentNode == null) break;
        }
        /////

        return currentNode;
    }

    public SimpleMDAGNode transition(SimpleMDAGNode[] mdagDataArray, char[] str, int offset)
    {
        SimpleMDAGNode currentNode = this;
        int numberOfChars = str.length - offset;

        //Iteratively _transition through the MDAG using the chars in str
        for (int i = 0; i < numberOfChars; i++)
        {
            currentNode = currentNode.transition(mdagDataArray, str[offset + i]);
            if (currentNode == null) break;
        }
        /////

        return currentNode;
    }


    /**
     * Follows a _transition path starting from the source node of a MDAG.
     *
     * @param mdagDataArray the array containing the data of the MDAG to be traversed
     * @param sourceNode    the dummy SimpleMDAGNode which functions as the source of the MDAG data in {@code mdagDataArray}
     * @param str           a String corresponding to a _transition path in the to-be-traversed MDAG
     * @return the SimpleMDAGNode at the end of the _transition path corresponding to
     * {@code str}, or null if such a _transition path is not present in the MDAG
     */
    public static SimpleMDAGNode traverseMDAG(SimpleMDAGNode[] mdagDataArray, SimpleMDAGNode sourceNode, String str)
    {
//        char firstLetter = str.charAt(0);

        //Loop through the SimpleMDAGNodes in the processing MDAG's source node's _transition set,
        //searching for the the one with a letter (char) equal to the first char of str.
        //We can use that target node to _transition through the MDAG with the rest of the string
        return sourceNode.transition(mdagDataArray, str.toCharArray());
//        for(int i = 0; i < sourceNode.transitionSetSize; i++)
//        {
//            if(mdagDataArray[i].getLetter() == firstLetter)
//                return mdagDataArray[i]._transition(mdagDataArray, str.substring(1));
//        }
//        /////
//
//        return null;
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder("SimpleMDAGNode{");
        sb.append("letter=").append(letter);
        sb.append(", isAcceptNode=").append(isAcceptNode);
        sb.append(", transitionSetSize=").append(transitionSetSize);
        sb.append(", transitionSetBeginIndex=").append(transitionSetBeginIndex);
        sb.append('}');
        return sb.toString();
    }

    @Override
    public void save(DataOutputStream out) throws Exception
    {
        out.writeChar(letter);
        out.writeByte(isAcceptNode ? 1 : 0);
        out.writeInt(transitionSetBeginIndex);
        out.writeInt(transitionSetSize);
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        letter = byteArray.nextChar();
        isAcceptNode = byteArray.nextByte() == 1;
        transitionSetBeginIndex = byteArray.nextInt();
        transitionSetSize = byteArray.nextInt();
        return true;
    }
}