package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.util.CorpusUtil;

public class TestNatureDictionaryMaker
{

    public static void main(String[] args)
    {
//        makeCoreDictionary("D:\\JavaProjects\\CorpusToolBox\\data\\2014", "data/dictionary/CoreNatureDictionary.txt");
//        EasyDictionary dictionary = EasyDictionary.create("data/dictionary/CoreNatureDictionary.txt");
        final NatureDictionaryMaker dictionaryMaker = new NatureDictionaryMaker();
        CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                dictionaryMaker.compute(CorpusUtil.convert2CompatibleList(document.getSimpleSentenceList(false))); // 再打一遍不拆分的
                dictionaryMaker.compute(CorpusUtil.convert2CompatibleList(document.getSimpleSentenceList(true)));  // 先打一遍拆分的
            }
        });
        dictionaryMaker.saveTxtTo("data/test/CoreNatureDictionary");
    }
    
}
