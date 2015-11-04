/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 22:17</create-date>
 *
 * <copyright file="java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import com.hankcs.hanlp.dependency.nnparser.action.Action;
import com.hankcs.hanlp.dependency.nnparser.action.ActionFactory;
import com.hankcs.hanlp.dependency.nnparser.action.ActionUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 动作转移系统
 * @author hankcs
 */
public class TransitionSystem
{
    /**
     * 所有依存关系的数量
     */
    int L;
    /**
     * 根节点label对应的id
     */
    int R;
    int D;

    public TransitionSystem()
    {
        L = 0;
        R = -1;
        D = -1;
    }

    /**
     * 设置根节点label对应的id
     * @param r
     */
    void set_root_relation(int r)
    {
        R = r;
    }

    /**
     * 设置所有依存关系的数量
     * @param l
     */
    void set_number_of_relations(int l)
    {
        L = l;
    }

    /**
     * 获取当前状态可能的动作（动作=shift | left | right + 依存关系，也就是说是一条既有方向又有依存关系名称的依存边）
     * @param source 当前状态
     * @param actions 输出可能动作
     */
    void get_possible_actions(State source,
                              List<Action> actions)
    {
        if (0 == L || -1 == R)
        {
            System.err.println("decoder: not initialized, please check if the root dependency relation is correct set by --root.");
            return;
        }
        actions.clear();

        if (!source.buffer_empty())
        {
            actions.add(ActionFactory.make_shift());
        }

        if (source.stack_size() == 2)
        {
            if (source.buffer_empty())
            {
                actions.add(ActionFactory.make_right_arc(R));
            }
        }
        else if (source.stack_size() > 2)
        {
            for (int l = 0; l < L; ++l)
            {
                if (l == R)
                {
                    continue;
                }
                actions.add(ActionFactory.make_left_arc(l));
                actions.add(ActionFactory.make_right_arc(l));
            }
        }
    }

    /**
     * 转移状态
     * @param source 源状态
     * @param act 动作
     * @param target 目标状态
     */
    void transit(State source, Action act, State target)
    {
        int deprel = 0;
        int[] deprel_inference = new int[]{deprel};
        if (ActionUtils.is_shift(act))
        {
            target.shift(source);
        }
        else if (ActionUtils.is_left_arc(act, deprel_inference))
        {
            deprel = deprel_inference[0];
            target.left_arc(source, deprel);
        }
        else if (ActionUtils.is_right_arc(act, deprel_inference))
        {
            deprel = deprel_inference[0];
            target.right_arc(source, deprel);
        }
        else
        {
            System.err.printf("unknown transition in transit: %d-%d", act.name(), act.rel());
        }
    }

    List<Integer> transform(List<Action> actions)
    {
        List<Integer> classes = new ArrayList<Integer>();
        transform(actions, classes);
        return classes;
    }

    void transform(List<Action> actions,
                   List<Integer> classes)
    {
        classes.clear();
        for (int i = 0; i < actions.size(); ++i)
        {
            classes.add(transform(actions.get(i)));
        }
    }

    /**
     * 转换动作为动作id
     * @param act 动作
     * @return 动作类型的依存关系id
     */
    int transform(Action act)
    {
        int deprel = 0;
        int[] deprel_inference = new int[]{deprel};
        if (ActionUtils.is_shift(act))
        {
            return 0;
        }
        else if (ActionUtils.is_left_arc(act, deprel_inference))
        {
            deprel = deprel_inference[0];
            return 1 + deprel;
        }
        else if (ActionUtils.is_right_arc(act, deprel_inference))
        {
            deprel = deprel_inference[0];
            return L + 1 + deprel;
        }
        else
        {
            System.err.printf("unknown transition in transform(Action): %d-%d", act.name(), act.rel());
        }
        return -1;
    }

    /**
     * 转换动作id为动作
     * @param act 动作类型的依存关系id
     * @return 动作
     */
    Action transform(int act)
    {
        if (act == 0)
        {
            return ActionFactory.make_shift();
        }
        else if (act < 1 + L)
        {
            return ActionFactory.make_left_arc(act - 1);
        }
        else if (act < 1 + 2 * L)
        {
            return ActionFactory.make_right_arc(act - 1 - L);
        }
        else
        {
            System.err.printf("unknown transition in transform(int): %d", act);
        }
        return new Action();
    }

    int number_of_transitions()
    {
        return L * 2 + 1;
    }
}
