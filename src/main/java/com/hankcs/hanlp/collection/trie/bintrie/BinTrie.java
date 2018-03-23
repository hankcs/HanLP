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

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 首字直接分配内存，之后二分动态数组的Trie树，能够平衡时间和空间
 *
 * @author hankcs
 */
public class BinTrie<V> extends BaseNode<V> implements ITrie<V>, Externalizable
{
    private int size;

    public BinTrie()
    {
        child = new BaseNode[65535 + 1];    // (int)Character.MAX_VALUE
        size = 0;
        status = Status.NOT_WORD_1;
    }

    /**
     * 插入一个词
     *
     * @param key
     * @param value
     */
    public void put(String key, V value)
    {
        if (key.length() == 0) return;  // 安全起见
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

    public void put(char[] key, V value)
    {
        BaseNode branch = this;
        for (int i = 0; i < key.length - 1; ++i)
        {
            // 除了最后一个字外，都是继续
            branch.addChild(new Node(key[i], Status.NOT_WORD_1, null));
            branch = branch.getChild(key[i]);
        }
        // 最后一个字加入时属性为end
        if (branch.addChild(new Node<V>(key[key.length - 1], Status.WORD_END_3, value)))
        {
            ++size; // 维护size
        }
    }

    /**
     * 设置键值对，当键不存在的时候会自动插入
     * @param key
     * @param value
     */
    public void set(String key, V value)
    {
        put(key.toCharArray(), value);
    }

    /**
     * 删除一个词
     *
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
        if (branch == null) return;
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

    public V get(char[] key)
    {
        BaseNode branch = this;
        for (char aChar : key)
        {
            if (branch == null) return null;
            branch = branch.getChild(aChar);
        }

        if (branch == null) return null;
        // 下面这句可以保证只有成词的节点被返回
        if (!(branch.status == Status.WORD_END_3 || branch.status == Status.WORD_MIDDLE_2)) return null;
        return (V) branch.getValue();
    }

    @Override
    public V[] getValueArray(V[] a)
    {
        if (a.length < size)
            a = (V[]) java.lang.reflect.Array.newInstance(
                    a.getClass().getComponentType(), size);
        int i = 0;
        for (Map.Entry<String, V> entry : entrySet())
        {
            a[i++] = entry.getValue();
        }
        return a;
    }

    /**
     * 获取键值对集合
     *
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
     * 键集合
     * @return
     */
    public Set<String> keySet()
    {
        TreeSet<String> keySet = new TreeSet<String>();
        for (Map.Entry<String, V> entry : entrySet())
        {
            keySet.add(entry.getKey());
        }

        return keySet;
    }

    /**
     * 前缀查询
     *
     * @param key 查询串
     * @return 键值对
     */
    public Set<Map.Entry<String, V>> prefixSearch(String key)
    {
        Set<Map.Entry<String, V>> entrySet = new TreeSet<Map.Entry<String, V>>();
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
     *
     * @param chars 字符串的字符数组
     * @param begin 开始的下标
     * @return
     */
    public LinkedList<Map.Entry<String, V>> commonPrefixSearchWithValue(char[] chars, int begin)
    {
        LinkedList<Map.Entry<String, V>> result = new LinkedList<Map.Entry<String, V>>();
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
                result.add(new AbstractMap.SimpleEntry<String, V>(sb.toString(), (V) branch.value));
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

    public boolean save(String path)
    {
        try
        {
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(path));
            for (BaseNode node : child)
            {
                if (node == null)
                {
                    out.writeInt(0);
                }
                else
                {
                    out.writeInt(1);
                    node.walkToSave(out);
                }
            }
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("保存到" + path + "失败" + TextUtility.exceptionToString(e));
            return false;
        }

        return true;
    }

    @Override
    public int build(TreeMap<String, V> keyValueMap)
    {
        for (Map.Entry<String, V> entry : keyValueMap.entrySet())
        {
            put(entry.getKey(), entry.getValue());
        }
        return 0;
    }

    /**
     * 保存到二进制输出流
     *
     * @param out
     * @return
     */
    public boolean save(DataOutputStream out)
    {
        try
        {
            for (BaseNode node : child)
            {
                if (node == null)
                {
                    out.writeInt(0);
                }
                else
                {
                    out.writeInt(1);
                    node.walkToSave(out);
                }
            }
        }
        catch (Exception e)
        {
            logger.warning("保存到" + out + "失败" + TextUtility.exceptionToString(e));
            return false;
        }

        return true;
    }

    /**
     * 从磁盘加载二分数组树
     *
     * @param path  路径
     * @param value 额外提供的值数组，按照值的字典序。（之所以要求提供它，是因为泛型的保存不归树管理）
     * @return 是否成功
     */
    public boolean load(String path, V[] value)
    {
        byte[] bytes = IOUtil.readBytes(path);
        if (bytes == null) return false;
        _ValueArray valueArray = new _ValueArray(value);
        ByteArray byteArray = new ByteArray(bytes);
        for (int i = 0; i < child.length; ++i)
        {
            int flag = byteArray.nextInt();
            if (flag == 1)
            {
                child[i] = new Node<V>();
                child[i].walkToLoad(byteArray, valueArray);
            }
        }
        size = value.length;

        return true;
    }

    /**
     * 只加载值，此时相当于一个set
     *
     * @param path
     * @return
     */
    public boolean load(String path)
    {
        byte[] bytes = IOUtil.readBytes(path);
        if (bytes == null) return false;
        _ValueArray valueArray = new _EmptyValueArray();
        ByteArray byteArray = new ByteArray(bytes);
        for (int i = 0; i < child.length; ++i)
        {
            int flag = byteArray.nextInt();
            if (flag == 1)
            {
                child[i] = new Node<V>();
                child[i].walkToLoad(byteArray, valueArray);
            }
        }
        size = -1;  // 不知道有多少

        return true;
    }

    public boolean load(ByteArray byteArray, _ValueArray valueArray)
    {
        for (int i = 0; i < child.length; ++i)
        {
            int flag = byteArray.nextInt();
            if (flag == 1)
            {
                child[i] = new Node<V>();
                child[i].walkToLoad(byteArray, valueArray);
            }
        }
        size = valueArray.value.length;

        return true;
    }

    public boolean load(ByteArray byteArray, V[] value)
    {
        return load(byteArray, newValueArray().setValue(value));
    }

    public _ValueArray newValueArray()
    {
        return new _ValueArray();
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException
    {
        out.writeInt(size);
        for (BaseNode node : child)
        {
            if (node == null)
            {
                out.writeInt(0);
            }
            else
            {
                out.writeInt(1);
                node.walkToSave(out);
            }
        }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException
    {
        size = in.readInt();
        for (int i = 0; i < child.length; ++i)
        {
            int flag = in.readInt();
            if (flag == 1)
            {
                child[i] = new Node<V>();
                child[i].walkToLoad(in);
            }
        }
    }

    /**
     * 最长匹配
     *
     * @param text      文本
     * @param processor 处理器
     */
    public void parseLongestText(String text, AhoCorasickDoubleArrayTrie.IHit<V> processor)
    {
        int length = text.length();
        for (int i = 0; i < length; ++i)
        {
            BaseNode<V> state = transition(text.charAt(i));
            if (state != null)
            {
                int to = i + 1;
                int end = to;
                V value = state.getValue();
                for (; to < length; ++to)
                {
                    state = state.transition(text.charAt(to));
                    if (state == null) break;
                    if (state.getValue() != null)
                    {
                        value = state.getValue();
                        end = to + 1;
                    }
                }
                if (value != null)
                {
                    processor.hit(i, end, value);
                    i = end - 1;
                }
            }
        }
    }

    /**
     * 最长匹配
     *
     * @param text      文本
     * @param processor 处理器
     */
    public void parseLongestText(char[] text, AhoCorasickDoubleArrayTrie.IHit<V> processor)
    {
        int length = text.length;
        for (int i = 0; i < length; ++i)
        {
            BaseNode<V> state = transition(text[i]);
            if (state != null)
            {
                int to = i + 1;
                int end = to;
                V value = state.getValue();
                for (; to < length; ++to)
                {
                    state = state.transition(text[to]);
                    if (state == null) break;
                    if (state.getValue() != null)
                    {
                        value = state.getValue();
                        end = to + 1;
                    }
                }
                if (value != null)
                {
                    processor.hit(i, end, value);
                    i = end - 1;
                }
            }
        }
    }

    /**
     * 匹配文本
     *
     * @param text      文本
     * @param processor 处理器
     */
    public void parseText(String text, AhoCorasickDoubleArrayTrie.IHit<V> processor)
    {
        int length = text.length();
        int begin = 0;
        BaseNode<V> state = this;

        for (int i = begin; i < length; ++i)
        {
            state = state.transition(text.charAt(i));
            if (state != null)
            {
                V value = state.getValue();
                if (value != null)
                {
                    processor.hit(begin, i + 1, value);
                }
            }
            else
            {
                i = begin;
                ++begin;
                state = this;
            }
        }
    }

    /**
     * 匹配文本
     *
     * @param text      文本
     * @param processor 处理器
     */
    public void parseText(char[] text, AhoCorasickDoubleArrayTrie.IHit<V> processor)
    {
        int length = text.length;
        int begin = 0;
        BaseNode<V> state = this;

        for (int i = begin; i < length; ++i)
        {
            state = state.transition(text[i]);
            if (state != null)
            {
                V value = state.getValue();
                if (value != null)
                {
                    processor.hit(begin, i + 1, value);
                }
            }
            else
            {
                i = begin;
                ++begin;
                state = this;
            }
        }
    }
}
