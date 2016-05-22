package com.hankcs.test.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import junit.framework.TestCase;

public class TestCoreDictionary extends TestCase
{
    public static void main(String[] args)
    {
//        BaseSearcher searcher = CoreDictionary.getSearcher("我是一个程序员");
//        Map.Entry entry;
//        while ((entry = searcher.next()) != null)
//        {
//            System.out.print(searcher.getOffset());
//            System.out.println(entry);
//        }
//        DictionaryMaker dictionaryMaker = DictionaryMaker.combineWhenNotInclude(new String[]{"data/dictionary/CoreNatureDictionary.txt", "data/dictionary/CoreNatureDictionary.mini.txt"});
//        dictionaryMaker.saveTxtTo("data/dictionary/CoreNatureDictionary.txt");
    }

    public void testLoad() throws Exception
    {
        HanLP.Config.enableDebug();
        CoreDictionary.get("速度");
    }
}
