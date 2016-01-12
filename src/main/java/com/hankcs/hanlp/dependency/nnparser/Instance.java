/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/30 19:29</create-date>
 *
 * <copyright file="Instance.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hankcs
 */
public class Instance
{
    List<String> raw_forms; //! The original form.
    List<String> forms;     //! The converted form.
    List<String> lemmas;    //! The lemmas.
    List<String> postags;   //! The postags.
    List<String> cpostags;  //! The cpostags.

    List<Integer> heads;
    List<Integer> deprelsidx;
    List<String> deprels;
    List<Integer> predict_heads;
    List<Integer> predict_deprelsidx;
    List<String> predict_deprels;

    public Instance()
    {
        forms = new ArrayList<String>();
        postags = new ArrayList<String>();
    }

    int size()
    {
        return forms.size();
    }

    boolean is_tree()
    {
        List<List<Integer>> tree = new ArrayList<List<Integer>>(heads.size());
        int root = -1;
        for (int modifier = 0; modifier < heads.size(); ++modifier)
        {
            int head = heads.get(modifier);
            if (head == -1)
            {
                root = modifier;
            }
            else
            {
                tree.get(head).add(modifier);
            }
        }
        boolean visited[] = new boolean[heads.size()];
        if (!is_tree_travel(root, tree, visited))
        {
            return false;
        }
        for (int i = 0; i < visited.length; ++i)
        {
            boolean visit = visited[i];
            if (!visit)
            {
                return false;
            }
        }
        return true;
    }

    boolean is_tree_travel(int now, List<List<Integer>> tree, boolean visited[])
    {
        if (visited[now])
        {
            return false;
        }
        visited[now] = true;
        for (int c = 0; c < tree.get(now).size(); ++c)
        {
            int next = tree.get(now).get(c);
            if (!is_tree_travel(next, tree, visited))
            {
                return false;
            }
        }
        return true;
    }

    boolean is_projective()
    {
        return !is_non_projective();
    }

    boolean is_non_projective()
    {
        for (int modifier = 0; modifier < heads.size(); ++modifier)
        {
            int head = heads.get(modifier);
            if (head < modifier)
            {
                for (int from = head + 1; from < modifier; ++from)
                {
                    int to = heads.get(from);
                    if (to < head || to > modifier)
                    {
                        return true;
                    }
                }
            }
            else
            {
                for (int from = modifier + 1; from < head; ++from)
                {
                    int to = heads.get(from);
                    if (to < modifier || to > head)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
