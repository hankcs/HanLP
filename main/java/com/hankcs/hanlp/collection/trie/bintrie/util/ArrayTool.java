/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/3 12:32</create-date>
 *
 * <copyright file="ArrayTool.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie.util;


import com.hankcs.hanlp.collection.trie.bintrie.BaseNode;

/**
 * @author He Han
 */
public class ArrayTool
{
    /**
     * 二分查找
     * @param branches 数组
     * @param node 要查找的node
     * @return 数组下标，小于0表示没找到
     */
    public static int binarySearch(BaseNode[] branches, BaseNode node)
    {
        int high = branches.length - 1;
        if (branches.length < 1)
        {
            return high;
        }
        int low = 0;
        while (low <= high)
        {
            int mid = (low + high) >>> 1;
            int cmp = branches[mid].compareTo(node);

            if (cmp < 0)
                low = mid + 1;
            else if (cmp > 0)
                high = mid - 1;
            else
                return mid;
        }
        return -(low + 1);
    }

    public static int binarySearch(BaseNode[] branches, char node)
    {
        int high = branches.length - 1;
        if (branches.length < 1)
        {
            return high;
        }
        int low = 0;
        while (low <= high)
        {
            int mid = (low + high) >>> 1;
            int cmp = branches[mid].compareTo(node);

            if (cmp < 0)
                low = mid + 1;
            else if (cmp > 0)
                high = mid - 1;
            else
                return mid;
        }
        return -(low + 1);
    }
}
