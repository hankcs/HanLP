/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/20 22:10</create-date>
 *
 * <copyright file="MDAGSet.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.MDAG;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * 基于MDAG（又称DAWG，Minimal Acyclic Finite-State Automata）的String Set
 *
 * @author hankcs
 */
public class MDAGSet extends MDAG implements Set<String>
{

    public MDAGSet(File dataFile) throws IOException
    {
        super(dataFile);
    }

    public MDAGSet(Collection<String> strCollection)
    {
        super(strCollection);
    }

    public MDAGSet()
    {
    }

    public MDAGSet(String dictionaryPath) throws IOException
    {
        super(dictionaryPath);
    }

    @Override
    public int size()
    {
        return getAllStrings().size();
    }

    @Override
    public boolean isEmpty()
    {
        return this.equivalenceClassMDAGNodeHashMap.size() != 0;
    }

    @Override
    public boolean contains(Object o)
    {
        if (o.getClass() != String.class) return false;
        return contains((String) o);
    }

    @Override
    public Iterator<String> iterator()
    {
        return getAllStrings().iterator();
    }

    @Override
    public Object[] toArray()
    {
        return getAllStrings().toArray();
    }

    @Override
    public <T> T[] toArray(T[] a)
    {
        return getAllStrings().toArray(a);
    }

    @Override
    public boolean add(String s)
    {
        addString(s);
        return true;
    }

    @Override
    public boolean remove(Object o)
    {
        if (o.getClass() == String.class)
        {
            removeString((String) o);
        }
        else
        {
            removeString(o.toString());
        }
        return true;
    }

    @Override
    public boolean containsAll(Collection<?> c)
    {
        for (Object e : c)
            if (!contains(e))
                return false;
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends String> c)
    {
        boolean modified = false;
        for (String e : c)
            if (add(e))
                modified = true;
        return modified;
    }

    @Override
    public boolean retainAll(Collection<?> c)
    {
        boolean modified = false;
        Iterator<String> it = iterator();
        while (it.hasNext())
        {
            if (!c.contains(it.next()))
            {
                it.remove();
                modified = true;
            }
        }
        return modified;
    }

    @Override
    public boolean removeAll(Collection<?> c)
    {
        boolean modified = false;
        Iterator<?> it = iterator();
        while (it.hasNext())
        {
            if (c.contains(it.next()))
            {
                it.remove();
                modified = true;
            }
        }
        return modified;
    }

    @Override
    public void clear()
    {
        sourceNode = new MDAGNode(false);
        simplifiedSourceNode = null;
        equivalenceClassMDAGNodeHashMap.clear();
        mdagDataArray = null;
        charTreeSet.clear();
        transitionCount = 0;
    }
}
