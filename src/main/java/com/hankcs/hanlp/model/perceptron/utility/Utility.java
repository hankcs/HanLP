/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM7:40</create-date>
 *
 * <copyright file="Utility.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.utility;

import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * @author hankcs
 */
public class Utility
{
    public static double[] prf(int[] stat)
    {
        return prf(stat[0], stat[1], stat[2]);
    }

    public static double[] prf(int goldTotal, int predTotal, int correct)
    {
        double precision = (correct * 100.0) / predTotal;
        double recall = (correct * 100.0) / goldTotal;
        double[] performance = new double[3];
        performance[0] = precision;
        performance[1] = recall;
        performance[2] = (2 * precision * recall) / (precision + recall);
        return performance;
    }

    /**
     * Fisher–Yates shuffle
     *
     * @param ar
     */
    public static void shuffleArray(int[] ar)
    {
        Random rnd = new Random();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    public static void shuffleArray(Instance[] ar)
    {
        Random rnd = new Random();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            Instance a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    public static String normalize(String text)
    {
        return text;
//        StringBuilder sb = new StringBuilder(text.length());
//        for (int i = 0; i < text.length(); i++)
//        {
//            sb.append(CharTable.convertPKUtoCWS(text.charAt(i)));
//        }
//        return sb.toString();
    }

    /**
     * 将人民日报格式的分词语料转化为空格分割的语料
     *
     * @param inputFolder 输入人民日报语料的上级目录(该目录下的所有文件都是一篇人民日报分词文章)
     * @param outputFile  输出一整个CRF训练格式的语料
     * @param begin       取多少个文档之后
     * @param end
     * @throws IOException 转换过程中的IO异常
     */
    public static void convertPKUtoCWS(String inputFolder, String outputFile, final int begin, final int end) throws IOException
    {
        final BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"));
        CorpusLoader.walk(inputFolder, new CorpusLoader.Handler()
                          {
                              int doc = 0;

                              @Override
                              public void handle(Document document)
                              {
                                  ++doc;
                                  if (doc < begin || doc > end) return;
                                  try
                                  {
                                      List<List<Word>> sentenceList = convertComplexWordToSimpleWord(document.getComplexSentenceList());
                                      if (sentenceList.size() == 0) return;
                                      for (List<Word> sentence : sentenceList)
                                      {
                                          if (sentence.size() == 0) continue;
                                          int index = 0;
                                          for (IWord iWord : sentence)
                                          {
                                              bw.write(iWord.getValue());
                                              if (++index != sentence.size())
                                              {
                                                  bw.write(' ');
                                              }
                                          }
                                          bw.newLine();
                                      }
                                  }
                                  catch (IOException e)
                                  {
                                      e.printStackTrace();
                                  }
                              }
                          }

        );
        bw.close();
    }


    /**
     * 将人民日报格式的分词语料转化为空格分割的语料
     *
     * @param inputFolder 输入人民日报语料的上级目录(该目录下的所有文件都是一篇人民日报分词文章)
     * @param outputFile  输出一整个CRF训练格式的语料
     * @param begin       取多少个文档之后
     * @param end
     * @throws IOException 转换过程中的IO异常
     */
    public static void convertPKUtoPOS(String inputFolder, String outputFile, final int begin, final int end) throws IOException
    {
        final BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"));
        CorpusLoader.walk(inputFolder, new CorpusLoader.Handler()
                          {
                              int doc = 0;

                              @Override
                              public void handle(Document document)
                              {
                                  ++doc;
                                  if (doc < begin || doc > end) return;
                                  try
                                  {
                                      List<List<Word>> sentenceList = document.getSimpleSentenceList();
                                      if (sentenceList.size() == 0) return;
                                      for (List<Word> sentence : sentenceList)
                                      {
                                          if (sentence.size() == 0) continue;
                                          int index = 0;
                                          for (IWord iWord : sentence)
                                          {
                                              bw.write(iWord.toString());
                                              if (++index != sentence.size())
                                              {
                                                  bw.write(' ');
                                              }
                                          }
                                          bw.newLine();
                                      }
                                  }
                                  catch (IOException e)
                                  {
                                      e.printStackTrace();
                                  }
                              }
                          }

        );
        bw.close();
    }

    private static List<List<Word>> convertComplexWordToSimpleWord(List<List<IWord>> document)
    {
        String nerTag[] = new String[]{"nr", "ns", "nt"};
        List<List<Word>> output = new ArrayList<List<Word>>(document.size());
        for (List<IWord> sentence : document)
        {
            List<Word> s = new ArrayList<Word>(sentence.size());
            for (IWord iWord : sentence)
            {
                if (iWord instanceof Word)
                {
                    s.add((Word) iWord);
                }
                else if (isNer(iWord, nerTag))
                {
                    s.add(new Word(iWord.getValue(), iWord.getLabel()));
                }
                else
                {
                    for (Word word : ((CompoundWord) iWord).innerList)
                    {
                        isNer(word, nerTag);
                        s.add(word);
                    }
                }
            }
            output.add(s);
        }

        return output;
    }

    private static boolean isNer(IWord word, String nerTag[])
    {
        for (String tag : nerTag)
        {
            if (word.getLabel().startsWith(tag))
            {
                word.setLabel(tag);
                return true;
            }
        }

        return false;
    }

    public static List<Word> toSimpleWordList(Sentence sentence)
    {
        List<Word> wordList = new LinkedList<Word>();
        for (IWord word : sentence.wordList)
        {
            if (word instanceof CompoundWord)
            {
                wordList.addAll(((CompoundWord) word).innerList);
            }
            else
            {
                wordList.add((Word) word);
            }
        }

        return wordList;
    }

    public static String[] toWordArray(List<Word> wordList)
    {
        String[] wordArray = new String[wordList.size()];
        int i = -1;
        for (Word word : wordList)
        {
            wordArray[++i] = word.getValue();
        }

        return wordArray;
    }
}
