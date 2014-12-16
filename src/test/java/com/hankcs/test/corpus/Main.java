package com.hankcs.test.corpus;


import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;

import java.util.List;

public class Main
{
    static String documentListPath = "data/documentList.ser";
    static String sentenceListPath = "data/simpleList.ser";
    public static void main(String[] args)
    {
        List<List<IWord>> sentenceList = CorpusLoader.loadSentenceList(sentenceListPath);
        System.out.println(sentenceList.get(0));
    }

    public static void initCorpus()
    {
        List<List<IWord>> sentenceList = CorpusLoader.convert2SentenceList("D:\\Doc\\语料库\\2014");
        CorpusLoader.saveSentenceList(sentenceList, sentenceListPath);
    }
}
