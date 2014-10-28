/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/17 14:20</create-date>
 *
 * <copyright file="SuggesterEx.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest;


import java.util.*;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * @author hankcs
 */
public class Suggester implements ISuggester
{
    Map<IdVector, Set<String>> storage;

    public Suggester()
    {
        storage = new TreeMap<>();
    }

    @Override
    public void addSentence(String sentence)
    {
        IdVector idVector = new IdVector(sentence);
        if (idVector.idArrayList.size() == 0) return;
        Set<String> set = storage.get(idVector);
        if (set == null)
        {
            set = new TreeSet<>();
            storage.put(idVector, set);
        }
        set.add(sentence);
    }

    @Override
    public List<String> suggest(String key, int size)
    {
        List<String> resultList = new ArrayList<>(size);
        TreeMap<Double, Set<String>> result = new TreeMap<>(Collections.reverseOrder());
        IdVector idVector1 = new IdVector(key);
        if (idVector1.idArrayList.size() == 0) return resultList;
        for (Map.Entry<IdVector, Set<String>> entry : storage.entrySet())
        {
            IdVector idVector2 = entry.getKey();
            Double score = idVector1.similarity(idVector2);
            Set<String> value = result.get(score);
            if (value == null)
            {
                value = new TreeSet<>();
                result.put(score, value);
            }
//            value.addAll(entry.getValue());
            value.add(entry.getValue().iterator().next());
        }
        for (Map.Entry<Double, Set<String>> entry : result.entrySet())
        {
//            System.out.print(entry.getKey() + " ");
            for (String sentence : entry.getValue())
            {
                if (resultList.size() >= size) return resultList;
                resultList.add(sentence);
//                System.out.println(sentence);
            }
        }

        return resultList;
    }
}
