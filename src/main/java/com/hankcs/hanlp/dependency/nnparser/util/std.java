/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 21:26</create-date>
 *
 * <copyright file="std.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.util;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

/**
 * @author hankcs
 */
public class std
{
    public static <E> void fill(List<E> list, E value)
    {
        if (list == null) return;
        ListIterator<E> listIterator = list.listIterator();
        while (listIterator.hasNext()) listIterator.set(value);
    }

    public static <E> List<E> create(int size, E value)
    {
        List<E> list = new ArrayList<E>(size);
        for (int i = 0; i < size; i++)
        {
            list.add(value);
        }

        return list;
    }

    public static <E> E pop_back(List<E> list)
    {
        E back = list.get(list.size() - 1);
        list.remove(list.size() - 1);
        return back;
    }
}
