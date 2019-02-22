/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-04 上午11:10</create-date>
 *
 * <copyright file="LabeledAction.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.perceptron.transition.parser;

/**
 * @author hankcs
 */
public class LabeledAction
{
    public Action action;
    public int label;

    public LabeledAction(Action action, int label)
    {
        this.action = action;
        this.label = label;
    }

    public LabeledAction(final int actionCode, final int labelSize)
    {
        if (actionCode == Action.Shift.ordinal())
        {
            action = Action.Shift;
        }
        else if (actionCode == Action.Reduce.ordinal())
        {
            action = Action.Reduce;
        }
        else if (actionCode >= Action.RightArc.ordinal() + labelSize)
        {
            label = actionCode - (Action.RightArc.ordinal() + labelSize);
            action = Action.LeftArc;
        }
        else if (actionCode >= Action.RightArc.ordinal())
        {
            label = actionCode - Action.RightArc.ordinal();
            action = Action.RightArc;
        }
        else if (actionCode == Action.Unshift.ordinal())
        {
            action = Action.Unshift;
        }
    }
}
