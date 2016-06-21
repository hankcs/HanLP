/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/3 12:27</create-date>
 *
 * <copyright file="Node.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie;


import com.hankcs.hanlp.collection.trie.bintrie.util.ArrayTool;

/**
 * 深度大于等于2的子节点
 *
 * @author He Han
 */
public class Node<V> extends BaseNode
{
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
                        target.value = null;
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
            BaseNode newChild[] = new BaseNode[child.length + 1];
            int insert = -(index + 1);
            System.arraycopy(child, 0, newChild, 0, insert);
            System.arraycopy(child, insert, newChild, insert + 1, child.length - insert);
            newChild[insert] = node;
            child = newChild;
            add = true;
        }
        return add;
    }

    /**
     * @param c      节点的字符
     * @param status 节点状态
     * @param value  值
     */
    public Node(char c, Status status, V value)
    {
        this.c = c;
        this.status = status;
        this.value = value;
    }

    public Node()
    {
    }

    @Override
    public BaseNode getChild(char c)
    {
        if (child == null) return null;
        int index = ArrayTool.binarySearch(child, c);
        if (index < 0) return null;

        return child[index];
    }
}
