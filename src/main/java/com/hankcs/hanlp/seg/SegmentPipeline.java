/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-29 5:05 PM</create-date>
 *
 * <copyright file="SegmentPipeline.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.seg;

import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.pipe.Pipe;

import java.util.*;

/**
 * @author hankcs
 */
public class SegmentPipeline extends Segment implements Pipe<String, List<Term>>, List<Pipe<List<IWord>, List<IWord>>>
{
    Pipe<String, List<IWord>> first;
    Pipe<List<IWord>, List<Term>> last;
    List<Pipe<List<IWord>, List<IWord>>> pipeList;

    private SegmentPipeline(Pipe<String, List<IWord>> first, Pipe<List<IWord>, List<Term>> last)
    {
        this.first = first;
        this.last = last;
        pipeList = new ArrayList<Pipe<List<IWord>, List<IWord>>>();
    }

    public SegmentPipeline(final Segment delegate)
    {
        this(new Pipe<String, List<IWord>>()
             {
                 @Override
                 public List<IWord> flow(String input)
                 {
                     List<IWord> task = new LinkedList<IWord>();
                     task.add(new Word(input, null));
                     return task;
                 }
             },
             new Pipe<List<IWord>, List<Term>>()
             {
                 @Override
                 public List<Term> flow(List<IWord> input)
                 {
                     List<Term> output = new ArrayList<Term>(input.size());
                     for (IWord word : input)
                     {
                         if (word.getLabel() == null)
                         {
                             output.addAll(delegate.seg(word.getValue()));
                         }
                         else
                         {
                             output.add(new Term(word.getValue(), Nature.create(word.getLabel())));
                         }
                     }
                     return output;
                 }
             });
        config = delegate.config;
    }


    @Override
    protected List<Term> segSentence(char[] sentence)
    {
        return seg(new String(sentence));
    }

    @Override
    public List<Term> seg(String text)
    {
        return flow(text);
    }

    @Override
    public List<Term> flow(String input)
    {
        List<IWord> i = first.flow(input);
        for (Pipe<List<IWord>, List<IWord>> pipe : pipeList)
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
    public Iterator<Pipe<List<IWord>, List<IWord>>> iterator()
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
    public boolean add(Pipe<List<IWord>, List<IWord>> pipe)
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
    public boolean addAll(Collection<? extends Pipe<List<IWord>, List<IWord>>> c)
    {
        return pipeList.addAll(c);
    }

    @Override
    public boolean addAll(int index, Collection<? extends Pipe<List<IWord>, List<IWord>>> c)
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
    public Pipe<List<IWord>, List<IWord>> get(int index)
    {
        return pipeList.get(index);
    }

    @Override
    public Pipe<List<IWord>, List<IWord>> set(int index, Pipe<List<IWord>, List<IWord>> element)
    {
        return pipeList.set(index, element);
    }

    @Override
    public void add(int index, Pipe<List<IWord>, List<IWord>> element)
    {
        pipeList.add(index, element);
    }

    @Override
    public Pipe<List<IWord>, List<IWord>> remove(int index)
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
    public ListIterator<Pipe<List<IWord>, List<IWord>>> listIterator()
    {
        return pipeList.listIterator();
    }

    @Override
    public ListIterator<Pipe<List<IWord>, List<IWord>>> listIterator(int index)
    {
        return pipeList.listIterator(index);
    }

    @Override
    public List<Pipe<List<IWord>, List<IWord>>> subList(int fromIndex, int toIndex)
    {
        return pipeList.subList(fromIndex, toIndex);
    }
}
