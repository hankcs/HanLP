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

import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.*;

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

    public static int[] evaluateCWS(String developFile, final PerceptronSegmenter segmenter) throws IOException
    {
        // int goldTotal = 0, predTotal = 0, correct = 0;
        final int[] stat = new int[3];
        Arrays.fill(stat, 0);
        IOUtility.loadInstance(developFile, new InstanceHandler()
        {
            @Override
            public boolean process(Sentence sentence)
            {
                List<Word> wordList = sentence.toSimpleWordList();
                String[] wordArray = toWordArray(wordList);
                stat[0] += wordArray.length;
                String text = com.hankcs.hanlp.utility.TextUtility.combine(wordArray);
                String[] predArray = segmenter.segment(text).toArray(new String[0]);
                stat[1] += predArray.length;

                int goldIndex = 0, predIndex = 0;
                int goldLen = 0, predLen = 0;

                while (goldIndex < wordArray.length && predIndex < predArray.length)
                {
                    if (goldLen == predLen)
                    {
                        if (wordArray[goldIndex].equals(predArray[predIndex]))
                        {
                            stat[2]++;
                            goldLen += wordArray[goldIndex].length();
                            predLen += wordArray[goldIndex].length();
                            goldIndex++;
                            predIndex++;
                        }
                        else
                        {
                            goldLen += wordArray[goldIndex].length();
                            predLen += predArray[predIndex].length();
                            goldIndex++;
                            predIndex++;
                        }
                    }
                    else if (goldLen < predLen)
                    {
                        goldLen += wordArray[goldIndex].length();
                        goldIndex++;
                    }
                    else
                    {
                        predLen += predArray[predIndex].length();
                        predIndex++;
                    }
                }

                return false;
            }
        });
        return stat;
    }

    public static List<String[]> convertSentenceToNER(Sentence sentence, NERTagSet tagSet)
    {
        List<String[]> collector = new LinkedList<String[]>();
        Set<String> nerLabels = tagSet.nerLabels;
        for (IWord word : sentence.wordList)
        {
            if (word instanceof CompoundWord)
            {
                List<Word> wordList = ((CompoundWord) word).innerList;
                Word[] words = wordList.toArray(new Word[0]);

                if (nerLabels.contains(word.getLabel()))
                {
                    collector.add(new String[]{words[0].value, words[0].label, tagSet.B_TAG_PREFIX + word.getLabel()});
                    for (int i = 1; i < words.length - 1; i++)
                    {
                        collector.add(new String[]{words[i].value, words[i].label, tagSet.M_TAG_PREFIX + word.getLabel()});
                    }
                    collector.add(new String[]{words[words.length - 1].value, words[words.length - 1].label,
                        tagSet.E_TAG_PREFIX + word.getLabel()});
                }
                else
                {
                    for (Word w : words)
                    {
                        collector.add(new String[]{w.value, w.label, tagSet.O_TAG});
                    }
                }
            }
            else
            {
                if (nerLabels.contains(word.getLabel()))
                {
                    // 单个实体
                    collector.add(new String[]{word.getValue(), word.getLabel(), tagSet.S_TAG});
                }
                else
                {
                    collector.add(new String[]{word.getValue(), word.getLabel(), tagSet.O_TAG});
                }
            }
        }
        return collector;
    }

    public static void normalize(Sentence sentence)
    {
        for (IWord word : sentence.wordList)
        {
            if (word instanceof CompoundWord)
            {
                for (Word child : ((CompoundWord) word).innerList)
                {
                    child.setValue(CharTable.convert(child.getValue()));
                }
            }
            else
            {
                word.setValue(CharTable.convert(word.getValue()));
            }
        }
    }
}
