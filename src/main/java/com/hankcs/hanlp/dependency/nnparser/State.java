/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 21:20</create-date>
 *
 * <copyright file="State.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import com.hankcs.hanlp.dependency.nnparser.action.Action;
import com.hankcs.hanlp.dependency.nnparser.action.ActionFactory;
import com.hankcs.hanlp.dependency.nnparser.util.std;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hankcs
 */
public class State
{
    //! The pointer to the previous state.
    /**
     * 栈
     */
    List<Integer> stack;
    /**
     * 队列的队首元素（的下标）
     */
    int buffer;               //! The front word in the buffer.
    /**
     * 上一个状态
     */
    State previous;    //! The pointer to the previous state.
    Dependency ref;    //! The pointer to the dependency tree.
    double score;             //! The score.
    /**
     * 上一次动作
     */
    Action last_action;       //! The last action.

    /**
     * 栈顶元素
     */
    int top0;                 //! The top word on the stack.
    /**
     * 栈顶元素的下一个元素（全栈第二个元素）
     */
    int top1;                 //! The second top word on the stack.
    List<Integer> heads;   //! Use to record the heads in current state.
    List<Integer> deprels; //! The dependency relation cached in state.
    /**
     * 当前节点的左孩子数量
     */
    List<Integer> nr_left_children;      //! The number of left children in this state.
    /**
     * 当前节点的右孩子数量
     */
    List<Integer> nr_right_children;     //! The number of right children in this state.
    List<Integer> left_most_child;       //! The left most child for each word in this state.
    List<Integer> right_most_child;      //! The right most child for each word in this state.
    List<Integer> left_2nd_most_child;   //! The left 2nd-most child for each word in this state.
    List<Integer> right_2nd_most_child;  //! The right 2nd-most child for each word in this state.

    public State()
    {
    }

    public State(Dependency ref)
    {
        this.ref = ref;
        stack = new ArrayList<Integer>(ref.size());
        clear();
        int L = ref.size();
        heads = std.create(L, -1);
        deprels = std.create(L, 0);
        nr_left_children = std.create(L, 0);
        nr_right_children = std.create(L, 0);
        left_most_child = std.create(L, -1);
        right_most_child = std.create(L, -1);
        left_2nd_most_child = std.create(L, -1);
        right_2nd_most_child = std.create(L, -1);
    }

    void clear()
    {
        score = 0;
        previous = null;
        top0 = -1;
        top1 = -1;
        buffer = 0;
        stack.clear();
        std.fill(heads, -1);
        std.fill(deprels, 0);
        std.fill(nr_left_children, 0);
        std.fill(nr_right_children, 0);
        std.fill(left_most_child, -1);
        std.fill(right_most_child, -1);
        std.fill(left_2nd_most_child, -1);
        std.fill(right_2nd_most_child, -1);
    }

    boolean can_shift()
    {
        return !buffer_empty();
    }

    boolean can_left_arc()
    {
        return stack_size() >= 2;
    }

    boolean can_right_arc()
    {
        return stack_size() >= 2;
    }

    /**
     * 克隆一个状态到自己
     * @param source 源状态
     */
    void copy(State source)
    {
        this.ref = source.ref;
        this.score = source.score;
        this.previous = source.previous;
        this.buffer = source.buffer;
        this.top0 = source.top0;
        this.top1 = source.top1;
        this.stack = source.stack;
        this.last_action = source.last_action;
        this.heads = source.heads;
        this.deprels = source.deprels;
        this.left_most_child = source.left_most_child;
        this.right_most_child = source.right_most_child;
        this.left_2nd_most_child = source.left_2nd_most_child;
        this.right_2nd_most_child = source.right_2nd_most_child;
        this.nr_left_children = source.nr_left_children;
        this.nr_right_children = source.nr_right_children;
    }

    /**
     * 更新栈的信息
     */
    void refresh_stack_information()
    {
        int sz = stack.size();
        if (0 == sz)
        {
            top0 = -1;
            top1 = -1;
        }
        else if (1 == sz)
        {
            top0 = stack.get(sz - 1);
            top1 = -1;
        }
        else
        {
            top0 = stack.get(sz - 1);
            top1 = stack.get(sz - 2);
        }
    }

    /**
     * 不建立依存关系，只转移句法分析的焦点，即原来的右焦点词变为新的左焦点词（本状态），依此类推。
     * @param source 右焦点词
     * @return 是否shift成功
     */
    boolean shift(State source)
    {
        if (!source.can_shift())
        {
            return false;
        }

        this.copy(source);
        stack.add(this.buffer);
        refresh_stack_information();
        ++this.buffer;

        this.last_action = ActionFactory.make_shift();
        this.previous = source;
        return true;
    }

    boolean left_arc(State source, int deprel)
    {
        if (!source.can_left_arc())
        {
            return false;
        }

        this.copy(source);
        stack.remove(stack.size() - 1);
        stack.set(stack.size() - 1, top0);

        heads.set(top1, top0);
        deprels.set(top1, deprel);

        if (-1 == left_most_child.get(top0))
        {
            // TP0 is left-isolate node.
            left_most_child.set(top0, top1);
        }
        else if (top1 < left_most_child.get(top0))
        {
            // (TP1, LM0, TP0)
            left_2nd_most_child.set(top0, left_most_child.get(top0));
            left_most_child.set(top0, top1);
        }
        else if (top1 < left_2nd_most_child.get(top0))
        {
            // (LM0, TP1, TP0)
            left_2nd_most_child.set(top0, top1);
        }

        nr_left_children.set(top0, nr_left_children.get(top0) + 1);
        refresh_stack_information();
        this.last_action = ActionFactory.make_left_arc(deprel);
        this.previous = source;
        return true;
    }

    boolean right_arc(State source, int deprel)
    {
        if (!source.can_right_arc())
        {
            return false;
        }

        this.copy(source);
        std.pop_back(stack);
        heads.set(top0, top1);
        deprels.set(top0, deprel);

        if (-1 == right_most_child.get(top1))
        {
            // TP1 is right-isolate node.
            right_most_child.set(top1, top0);
        }
        else if (right_most_child.get(top1) < top0)
        {
            right_2nd_most_child.set(top1, right_most_child.get(top1));
            right_most_child.set(top1, top0);
        }
        else if (right_2nd_most_child.get(top1) < top0)
        {
            right_2nd_most_child.set(top1, top0);
        }
        nr_right_children.set(top1, nr_right_children.get(top1) + 1);
        refresh_stack_information();
        this.last_action = ActionFactory.make_right_arc(deprel);
        this.previous = source;
        return true;
    }

    int cost(List<Integer> gold_heads,
             List<Integer> gold_deprels)
    {
        List<List<Integer>> tree = new ArrayList<List<Integer>>(gold_heads.size());
        for (int i = 0; i < gold_heads.size(); ++i)
        {
            int h = gold_heads.get(i);
            if (h >= 0)
            {
                tree.get(h).add(i);
            }
        }

        List<Integer> sigma_l = stack;
        List<Integer> sigma_r = new ArrayList<Integer>();
        sigma_r.add(stack.get(stack.size() - 1));

        boolean[] sigma_l_mask = new boolean[gold_heads.size()];
        boolean[] sigma_r_mask = new boolean[gold_heads.size()];
        for (int s = 0; s < sigma_l.size(); ++s)
        {
            sigma_l_mask[sigma_l.get(s)] = true;
        }

        for (int i = buffer; i < ref.size(); ++i)
        {
            if (gold_heads.get(i) < buffer)
            {
                sigma_r.add(i);
                sigma_r_mask[i] = true;
                continue;
            }

            List<Integer> node = tree.get(i);
            for (int d = 0; d < node.size(); ++d)
            {
                if (sigma_l_mask[node.get(d)] || sigma_r_mask[node.get(d)])
                {
                    sigma_r.add(i);
                    sigma_r_mask[i] = true;
                    break;
                }
            }
        }

        int len_l = sigma_l.size();
        int len_r = sigma_r.size();

        // typedef boost.multi_array<int, 3> array_t;
        // array_t T(boost.extents[len_l][len_r][len_l+len_r-1]);
        // std.fill( T.origin(), T.origin()+ T.num_elements(), 1024);
        int[][][] T = new int[len_l][len_r][len_l + len_r - 1];
        for (int[][] one : T)
        {
            for (int[] two : one)
            {
                for (int i = 0; i < two.length; i++)
                {
                    two[i] = 1024;
                }
            }
        }

        T[0][0][len_l - 1] = 0;
        for (int d = 0; d < len_l + len_r - 1; ++d)
        {
            for (int j = Math.max(0, d - len_l + 1); j < Math.min(d + 1, len_r); ++j)
            {
                int i = d - j;
                if (i < len_l - 1)
                {
                    int i_1 = sigma_l.get(len_l - i - 2);
                    int i_1_rank = len_l - i - 2;
                    for (int rank = len_l - i - 1; rank < len_l; ++rank)
                    {
                        int h = sigma_l.get(rank);
                        int h_rank = rank;
                        T[i + 1][j][h_rank] = Math.min(T[i + 1][j][h_rank],
                                                       T[i][j][h_rank] + (gold_heads.get(i_1) == h ? 0 : 2));
                        T[i + 1][j][i_1_rank] = Math.min(T[i + 1][j][i_1_rank],
                                                         T[i][j][h_rank] + (gold_heads.get(h) == i_1 ? 0 : 2));
                    }
                    for (int rank = 1; rank < j + 1; ++rank)
                    {
                        int h = sigma_r.get(rank);
                        int h_rank = len_l + rank - 1;
                        T[i + 1][j][h_rank] = Math.min(T[i + 1][j][h_rank],
                                                       T[i][j][h_rank] + (gold_heads.get(i_1) == h ? 0 : 2));
                        T[i + 1][j][i_1_rank] = Math.min(T[i + 1][j][i_1_rank],
                                                         T[i][j][h_rank] + (gold_heads.get(h) == i_1 ? 0 : 2));
                    }
                }
                if (j < len_r - 1)
                {
                    int j_1 = sigma_r.get(j + 1);
                    int j_1_rank = len_l + j;
                    for (int rank = len_l - i - 1; rank < len_l; ++rank)
                    {
                        int h = sigma_l.get(rank);
                        int h_rank = rank;
                        T[i][j + 1][h_rank] = Math.min(T[i][j + 1][h_rank],
                                                       T[i][j][h_rank] + (gold_heads.get(j_1) == h ? 0 : 2));
                        T[i][j + 1][j_1_rank] = Math.min(T[i][j + 1][j_1_rank],
                                                         T[i][j][h_rank] + (gold_heads.get(h) == j_1 ? 0 : 2));
                    }
                    for (int rank = 1; rank < j + 1; ++rank)
                    {
                        int h = sigma_r.get(rank);
                        int h_rank = len_l + rank - 1;
                        T[i][j + 1][h_rank] = Math.min(T[i][j + 1][h_rank],
                                                       T[i][j][h_rank] + (gold_heads.get(j_1) == h ? 0 : 2));
                        T[i][j + 1][j_1_rank] = Math.min(T[i][j + 1][j_1_rank],
                                                         T[i][j][h_rank] + (gold_heads.get(h) == j_1 ? 0 : 2));
                    }
                }
            }
        }
        int penalty = 0;
        for (int i = 0; i < buffer; ++i)
        {
            if (heads.get(i) != -1)
            {
                if (heads.get(i) != gold_heads.get(i))
                {
                    penalty += 2;
                }
                else if (deprels.get(i) != gold_deprels.get(i))
                {
                    penalty += 1;
                }
            }
        }
        return T[len_l - 1][len_r - 1][0] + penalty;
    }

    /**
     * 队列是否为空
     * @return
     */
    boolean buffer_empty()
    {
        return (this.buffer == this.ref.size());
    }

    /**
     * 栈的大小
     * @return
     */
    int stack_size()
    {
        return (this.stack.size());
    }
}
