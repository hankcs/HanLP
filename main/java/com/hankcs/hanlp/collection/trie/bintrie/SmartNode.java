/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 17:09</create-date>
 *
 * <copyright file="SmartNode.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie;

import com.hankcs.hanlp.collection.trie.bintrie.util.ArrayTool;

/**
 * @author hankcs
 */
public class SmartNode<V> extends BaseNode<V>
{
    /**
     * 动态平衡时间和空间的比率
     */
    final static double RATE = 0.9;
    /**
     * 超出此值即拓展子节点为65535的数组
     */
    final static int LIMIT = (int) (65535 * RATE);

    public SmartNode(char c, Status status, V value)
    {
        this.c = c;
        this.status = status;
        this.value = value;
    }

    public SmartNode()
    {
    }

    @Override
    protected boolean addChild(BaseNode node)
    {
        boolean add = false;
        if (child == null)
        {
            child = new BaseNode[0];
        }
        int index = ArrayTool.binarySearch(child, node);
        if (index >= 0)
        {
            BaseNode target = child[index];
            switch (node.status)
            {
                case UNDEFINED_0:
                    if (target.status != Status.NOT_WORD_1)
                    {
                        target.status = Status.NOT_WORD_1;
                        add = true;
                    }
                    break;
                case NOT_WORD_1:
                    if (target.status == Status.WORD_END_3)
                    {
                        target.status = Status.WORD_MIDDLE_2;
                    }
                    break;
                case WORD_END_3:
                    if (target.status != Status.WORD_END_3)
                    {
                        target.status = Status.WORD_MIDDLE_2;
                    }
                    if (target.getValue() == null)
                    {
                        add = true;
                    }
                    target.setValue(node.getValue());
                    break;
            }
        }
        else
        {
            // 如果数组内元素接近于最大值直接数组定位，rate是内存和速度的一个平衡
            if (child != null && child.length >= LIMIT)
            {
                BaseNode newChild[] = new BaseNode[65535];
                for (BaseNode b : child)
                {
                    newChild[b.c] = b;
                }
                newChild[node.c] = node;
                child = newChild;
            }
            else
            {
                BaseNode newChild[] = new BaseNode[child.length + 1];
                int insert = -(index + 1);
                System.arraycopy(child, 0, newChild, 0, insert);
                System.arraycopy(child, insert, newChild, insert + 1, child.length - insert);
                newChild[insert] = node;
                child = newChild;
            }
            add = true;
        }
        return add;
    }

    @Override
    public BaseNode getChild(char c)
    {
        if (child == null) return null;
        if (child.length == 65535)  // 已经被拓展为hash trie树，直接按c寻址返回
        {
            return child[c];
        }
        int index = ArrayTool.binarySearch(child, c);
        if (index < 0) return null;

        return child[index];
    }

    @Override
    protected BaseNode<V> newInstance()
    {
        return new SmartNode<>();
    }
}
