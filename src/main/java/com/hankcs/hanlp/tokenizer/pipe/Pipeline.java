/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-29 4:51 PM</create-date>
 *
 * <copyright file="Pipeline.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.pipe;

import java.util.*;

/**
 * 流水线
 *
 * @author hankcs
 */
public class Pipeline<I, M, O> implements Pipe<I, O>, List<Pipe<M, M>>
{
    /**
     * 入口
     */
    protected Pipe<I, M> first;
    /**
     * 出口
     */
    protected Pipe<M, O> last;
    /**
     * 中间部分
     */
    protected LinkedList<Pipe<M, M>> pipeList;

    public Pipeline(Pipe<I, M> first, Pipe<M, O> last)
    {
        this.first = first;
        this.last = last;
        pipeList = new LinkedList<Pipe<M, M>>();
    }

    @Override
    public O flow(I input)
    {
        M i = first.flow(input);
        for (Pipe<M, M> pipe : pipeList)
        {
            i = pipe.flow(i);
        }
        return last.flow(i);
    }

    @Override
    public int size()
    {
        return pipeList.size();
    }

    @Override
    public boolean isEmpty()
    {
        return pipeList.isEmpty();
    }

    @Override
    public boolean contains(Object o)
    {
        return pipeList.contains(o);
    }

    @Override
    public Iterator<Pipe<M, M>> iterator()
    {
        return pipeList.iterator();
    }

    @Override
    public Object[] toArray()
    {
        return pipeList.toArray();
    }

    @Override
    public <T> T[] toArray(T[] a)
    {
        return pipeList.toArray(a);
    }

    @Override
    public boolean add(Pipe<M, M> pipe)
    {
        return pipeList.add(pipe);
    }

    @Override
    public boolean remove(Object o)
    {
        return pipeList.remove(o);
    }

    @Override
    public boolean containsAll(Collection<?> c)
    {
        return pipeList.containsAll(c);
    }

    @Override
    public boolean addAll(Collection<? extends Pipe<M, M>> c)
    {
        return pipeList.addAll(c);
    }

    @Override
    public boolean addAll(int index, Collection<? extends Pipe<M, M>> c)
    {
        return pipeList.addAll(c);
    }

    @Override
    public boolean removeAll(Collection<?> c)
    {
        return pipeList.removeAll(c);
    }

    @Override
    public boolean retainAll(Collection<?> c)
    {
        return pipeList.retainAll(c);
    }

    @Override
    public void clear()
    {
        pipeList.clear();
    }

    @Override
    public boolean equals(Object o)
    {
        return pipeList.equals(o);
    }

    @Override
    public int hashCode()
    {
        return pipeList.hashCode();
    }

    @Override
    public Pipe<M, M> get(int index)
    {
        return pipeList.get(index);
    }

    @Override
    public Pipe<M, M> set(int index, Pipe<M, M> element)
    {
        return pipeList.set(index, element);
    }

    @Override
    public void add(int index, Pipe<M, M> element)
    {
        pipeList.add(index, element);
    }

    /**
     * 以最高优先级加入管道
     *
     * @param pipe
     */
    public void addFirst(Pipe<M, M> pipe)
    {
        pipeList.addFirst(pipe);
    }

    /**
     * 以最低优先级加入管道
     *
     * @param pipe
     */
    public void addLast(Pipe<M, M> pipe)
    {
        pipeList.addLast(pipe);
    }

    @Override
    public Pipe<M, M> remove(int index)
    {
        return pipeList.remove(index);
    }

    @Override
    public int indexOf(Object o)
    {
        return pipeList.indexOf(o);
    }

    @Override
    public int lastIndexOf(Object o)
    {
        return pipeList.lastIndexOf(o);
    }

    @Override
    public ListIterator<Pipe<M, M>> listIterator()
    {
        return pipeList.listIterator();
    }

    @Override
    public ListIterator<Pipe<M, M>> listIterator(int index)
    {
        return pipeList.listIterator(index);
    }

    @Override
    public List<Pipe<M, M>> subList(int fromIndex, int toIndex)
    {
        return pipeList.subList(fromIndex, toIndex);
    }
}