/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 20:52</create-date>
 *
 * <copyright file="ActionUtils.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.action;

import com.hankcs.hanlp.dependency.nnparser.Dependency;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hankcs
 */
public class ActionUtils implements ActionType
{
    public static boolean is_shift(final Action act)
    {
        return (act.name() == kShift);
    }

    public static boolean is_left_arc(final Action act, int[] deprel)
    {
        if (act.name() == kLeftArc)
        {
            deprel[0] = act.rel();
            return true;
        }
        deprel[0] = 0;
        return false;
    }

    public static boolean is_right_arc(final Action act, int[] deprel)
    {
        if (act.name() == kRightArc)
        {
            deprel[0] = act.rel();
            return true;
        }
        deprel[0] = 0;
        return false;
    }

    void get_oracle_actions(List<Integer> heads,
                            List<Integer> deprels,
                            List<Action> actions)
    {
        // The oracle finding algorithm for arcstandard is using a in-order tree
        // searching.
        int N = heads.size();
        int root = -1;
        List<List<Integer>> tree = new ArrayList<List<Integer>>(N);

        actions.clear();
        for (int i = 0; i < N; ++i)
        {
            int head = heads.get(i);
            if (head == -1)
            {
                if (root == -1)
                    System.err.println("error: there should be only one root.");
                root = i;
            }
            else
            {
                tree.get(head).add(i);
            }
        }

        get_oracle_actions_travel(root, heads, deprels, tree, actions);
    }

    void get_oracle_actions_travel(int root,
                                   List<Integer> heads,
                                   List<Integer> deprels,
                                   List<List<Integer>> tree,
                                   List<Action> actions)
    {
        List<Integer> children = tree.get(root);

        int i;
        for (i = 0; i < children.size() && children.get(i) < root; ++i)
        {
            get_oracle_actions_travel(children.get(i), heads, deprels, tree, actions);
        }

        actions.add(ActionFactory.make_shift());

        for (int j = i; j < children.size(); ++j)
        {
            int child = children.get(j);
            get_oracle_actions_travel(child, heads, deprels, tree, actions);
            actions.add(ActionFactory.make_right_arc (deprels.get(child)));
        }

        for (int j = i - 1; j >= 0; --j)
        {
            int child = children.get(j);
            actions.add(ActionFactory.make_left_arc (deprels.get(child)));
        }
    }

    void get_oracle_actions2( Dependency instance,
                        List<Action> actions)
    {
        get_oracle_actions2(instance.heads, instance.deprels, actions);
    }
    
    void get_oracle_actions2(List<Integer> heads,
                             List<Integer> deprels,
                             List<Action> actions) {
        actions.clear();
        int len = heads.size();
        List<Integer> sigma = new ArrayList<Integer>();
        int beta = 0;
        List<Integer> output = new ArrayList<Integer>(len);
        for (int i = 0; i < len; i++)
        {
            output.add(-1);
        }

        int step = 0;
        while (!(sigma.size() ==1 && beta == len))
        {
            int[] beta_reference = new int[]{beta};
            get_oracle_actions_onestep(heads, deprels, sigma, beta_reference, output, actions);
            beta = beta_reference[0];
        }
    }
    
    void get_oracle_actions_onestep(List<Integer> heads,
                                    List<Integer> deprels,
                                    List<Integer> sigma,
                                    int[] beta,
                                    List<Integer> output,
                                    List<Action> actions)
    {
        int top0 = (sigma.size() > 0 ? sigma.get(sigma.size() - 1) : -1);
        int top1 = (sigma.size() > 1 ? sigma.get(sigma.size() - 2) : -1);

        boolean all_descendents_reduced = true;
        if (top0 >= 0)
        {
            for (int i = 0; i < heads.size(); ++i)
            {
                if (heads.get(i) == top0 && output.get(i) != top0)
                {
                    // _INFO << i << " " << output[i];
                    all_descendents_reduced = false;
                    break;
                }
            }
        }

        if (top1 >= 0 && heads.get(top1) == top0)
        {
            actions.add(ActionFactory.make_left_arc(deprels.get(top1)));
            output.set(top1, top0);
            sigma.remove(sigma.size() - 1);
            sigma.set(sigma.size() - 1, top0);
        }
        else if (top1 >= 0 && heads.get(top0) == top1 && all_descendents_reduced)
        {
            actions.add(ActionFactory.make_right_arc(deprels.get(top0)));
            output.set(top0, top1);
            sigma.remove(sigma.size() - 1);
        }
        else if (beta[0] < heads.size())
        {
            actions.add(ActionFactory.make_shift ());
            sigma.add(beta[0]);
            ++beta[0];
        }
    }
}
