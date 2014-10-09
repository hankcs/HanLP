package com.hankcs.test.seg;

import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.dictionary.CustomDictionary;

import java.util.Map;

public class testCustomDictionary
{
    public static void main(String[] args)
    {
        BaseSearcher searcher = CustomDictionary.getSearcher("我是一个码农");
        Map.Entry entry;
        while ((entry = searcher.next()) != null)
        {
            System.out.println(entry);
        }
    }
}
