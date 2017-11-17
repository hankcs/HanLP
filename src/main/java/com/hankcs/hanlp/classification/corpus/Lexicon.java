/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/10 PM4:24</create-date>
 *
 * <copyright file="Lexicon.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.corpus;


import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * 词表
 *
 * @author hankcs
 */
public class Lexicon
{
    public BinTrie<Integer> wordId;
    public List<String> idWord;

    public Lexicon()
    {
        wordId = new BinTrie<Integer>();
        idWord = new LinkedList<String>();
    }

    public Lexicon(BinTrie<Integer> wordIdTrie)
    {
        wordId = wordIdTrie;
    }

    public int addWord(String word)
    {
        assert word != null;
        char[] charArray = word.toCharArray();
        Integer id = wordId.get(charArray);
        if (id == null)
        {
            id = wordId.size();
            wordId.put(charArray, id);
            idWord.add(word);
            assert idWord.size() == wordId.size();
        }

        return id;
    }

    public Integer getId(String word)
    {
        return wordId.get(word);
    }

    public String getWord(int id)
    {
        assert 0 <= id;
        assert id <= idWord.size();
        return idWord.get(id);
    }

    public int size()
    {
        return idWord.size();
    }

    public String[] getWordIdArray()
    {
        String[] wordIdArray = new String[idWord.size()];
        if (idWord.isEmpty()) return wordIdArray;
        int p = -1;
        Iterator<String> iterator = idWord.iterator();
        while (iterator.hasNext())
        {
            wordIdArray[++p] = iterator.next();
        }

        return wordIdArray;
    }
}