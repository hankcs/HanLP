/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/3 11:34</create-date>
 *
 * <copyright file="BinTrie.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie;

import java.util.*;

/**
 * 首字直接分配内存，之后二分动态数组的Trie树，能够平衡时间和空间
 * @author hankcs
 */
public class BinTrie<V> extends BaseNode<V>
{
    private int size;
    public BinTrie()
    {
        child = new BaseNode[65535];    // (int)Character.MAX_VALUE
        size = 0;
    }

    /**
     * 插入一个词
     *
     * @param key
     * @param value
     */
    public void put(String key, V value)
    {
        BaseNode branch = this;
        char[] chars = key.toCharArray();
        for (int i = 0; i < chars.length - 1; ++i)
        {
            // 除了最后一个字外，都是继续
            branch.addChild(new Node(chars[i], Status.NOT_WORD_1, null));
            branch = branch.getChild(chars[i]);
        }
        // 最后一个字加入时属性为end
        if (branch.addChild(new Node<V>(chars[chars.length - 1], Status.WORD_END_3, value)))
        {
            ++size; // 维护size
        }
    }

    /**
     * 删除一个词
     * @param key
     */
    public void remove(String key)
    {
        BaseNode branch = this;
        char[] chars = key.toCharArray();
        for (int i = 0; i < chars.length - 1; ++i)
        {
            if (branch == null) return;
            branch = branch.getChild(chars[i]);
        }
        // 最后一个字设为undefined
        if (branch.addChild(new Node(chars[chars.length - 1], Status.UNDEFINED_0, value)))
        {
            --size;
        }
    }

    public boolean containsKey(String key)
    {
        BaseNode branch = this;
        char[] chars = key.toCharArray();
        for (char aChar : chars)
        {
            if (branch == null) return false;
            branch = branch.getChild(aChar);
        }

        return branch != null && (branch.status == Status.WORD_END_3 || branch.status == Status.WORD_MIDDLE_2);
    }

    public V get(String key)
    {
        BaseNode branch = this;
        char[] chars = key.toCharArray();
        for (char aChar : chars)
        {
            if (branch == null) return null;
            branch = branch.getChild(aChar);
        }

        if (branch == null) return null;
        // 下面这句可以保证只有成词的节点被返回
        if (!(branch.status == Status.WORD_END_3 || branch.status == Status.WORD_MIDDLE_2)) return null;
        return (V) branch.getValue();
    }

    /**
     * 获取键值对集合
     * @return
     */
    public Set<Map.Entry<String, V>> entrySet()
    {
        Set<Map.Entry<String, V>> entrySet = new TreeSet<Map.Entry<String, V>>();
        StringBuilder sb = new StringBuilder();
        for (BaseNode node : child)
        {
            if (node == null) continue;
            node.walk(new StringBuilder(sb.toString()), entrySet);
        }
        return entrySet;
    }

    /**
     * 前缀查询
     * @param key 查询串
     * @return 键值对
     */
    public Set<Map.Entry<String, V>> prefixSearch(String key)
    {
        Set<Map.Entry<String, V>> entrySet = new TreeSet<>();
        StringBuilder sb = new StringBuilder(key.substring(0, key.length() - 1));
        BaseNode branch = this;
        char[] chars = key.toCharArray();
        for (char aChar : chars)
        {
            if (branch == null) return entrySet;
            branch = branch.getChild(aChar);
        }

        if (branch == null) return entrySet;
        branch.walk(sb, entrySet);
        return entrySet;
    }

    /**
     * 前缀查询，包含值
     *
     * @param key 键
     * @return 键值对列表
     */
    public LinkedList<Map.Entry<String, V>> commonPrefixSearchWithValue(String key)
    {
        char[] chars = key.toCharArray();
        return commonPrefixSearchWithValue(chars, 0);
    }

    /**
     * 前缀查询，通过字符数组来表示字符串可以优化运行速度
     * @param chars 字符串的字符数组
     * @param begin 开始的下标
     * @return
     */
    public LinkedList<Map.Entry<String, V>> commonPrefixSearchWithValue(char[] chars, int begin)
    {
        LinkedList<Map.Entry<String, V>> result = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        BaseNode branch = this;
        for (int i = begin; i < chars.length; ++i)
        {
            char aChar = chars[i];
            branch = branch.getChild(aChar);
            if (branch == null || branch.status == Status.UNDEFINED_0) return result;
            sb.append(aChar);
            if (branch.status == Status.WORD_MIDDLE_2 || branch.status == Status.WORD_END_3)
            {
                result.add(new AbstractMap.SimpleEntry<>(sb.toString(), (V) branch.value));
            }
        }

        return result;
    }

    @Override
    protected boolean addChild(BaseNode node)
    {
        boolean add = false;
        char c = node.getChar();
        BaseNode target = getChild(c);
        if (target == null)
        {
            child[c] = node;
            add = true;
        }
        else
        {
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
                    if (target.status == Status.NOT_WORD_1)
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
        return add;
    }

    public int size()
    {
        return size;
    }

    @Override
    protected char getChar()
    {
        return 0;   // 根节点没有char
    }

    @Override
    public BaseNode getChild(char c)
    {
        return child[c];
    }

}
