/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 20:48</create-date>
 *
 * <copyright file="Dependency.java" company="��ũ��">
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
public class Dependency
{
    public List<Integer> forms;
    public List<Integer> postags;
    public List<Integer> heads;
    public List<Integer> deprels;

    private static ArrayList<Integer> allocate()
    {
        return new ArrayList<Integer>();
    }

    public Dependency()
    {
        forms = allocate();
        postags = allocate();
        heads = allocate();
        deprels = allocate();
    }

    int size()
    {
        return forms.size();
    }
}
