/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-13 2:26 PM</create-date>
 *
 * <copyright file="Vocabulary.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.model.perceptron.common.IStringIdMap;

/**
 * @author hankcs
 */
public class Vocabulary implements IStringIdMap
{
    private BinTrie<Integer> trie;
    boolean mutable;
    private static final int UNK = 0;

    public Vocabulary(BinTrie<Integer> trie, boolean mutable)
    {
        this.trie = trie;
        this.mutable = mutable;
    }

    public Vocabulary()
    {
        this(new BinTrie<Integer>(), true);
        trie.put("\t", UNK);
    }

    @Override
    public int idOf(String string)
    {
        Integer id = trie.get(string);
        if (id == null)
        {
            if (mutable)
            {
                id = trie.size();
                trie.put(string, id);
            }
            else
                id = UNK;
        }
        return id;
    }
}
