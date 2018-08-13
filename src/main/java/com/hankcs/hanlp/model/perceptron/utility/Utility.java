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

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

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

    public static <T> void shuffleArray(T[] ar)
    {
        Random rnd = new Random();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            T a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    public static String normalize(String text)
    {
        return CharTable.convert(text);
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

    /**
     * 将句子转换为 （单词，词性，NER标签）三元组
     *
     * @param sentence
     * @param tagSet
     * @return
     */
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

    public static Map<String, double[]> evaluateNER(NERecognizer recognizer, String goldFile)
    {
        Map<String, double[]> scores = new TreeMap<String, double[]>();
        double[] avg = new double[]{0, 0, 0};
        scores.put("avg.", avg);
        NERTagSet tagSet = recognizer.getNERTagSet();
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(goldFile);
        for (String line : lineIterator)
        {
            line = line.trim();
            if (line.isEmpty()) continue;
            Sentence sentence = Sentence.create(line);
            if (sentence == null) continue;
            String[][] table = reshapeNER(convertSentenceToNER(sentence, tagSet));
            Set<String> pred = combineNER(recognizer.recognize(table[0], table[1]), tagSet);
            Set<String> gold = combineNER(table[2], tagSet);
            for (String p : pred)
            {
                String type = p.split("\t")[2];
                double[] s = scores.get(type);
                if (s == null)
                {
                    s = new double[]{0, 0, 0};
                    scores.put(type, s);
                }
                if (gold.contains(p))
                {
                    ++s[2]; // 正确识别该类命名实体数
                    ++avg[2];
                }
                ++s[0]; // 识别出该类命名实体总数
                ++avg[0];
            }
            for (String g : gold)
            {
                String type = g.split("\t")[2];
                double[] s = scores.get(type);
                if (s == null)
                {
                    s = new double[]{0, 0, 0};
                    scores.put(type, s);
                }
                ++s[1]; // 该类命名实体总数
                ++avg[1];
            }
        }
        for (double[] s : scores.values())
        {
            if (s[2] == 0)
            {
                s[0] = 0;
                s[1] = 0;
                continue;
            }
            s[1] = s[2] / s[1] * 100; // R=正确识别该类命名实体数/该类命名实体总数×100%
            s[0] = s[2] / s[0] * 100; // P=正确识别该类命名实体数/识别出该类命名实体总数×100%
            s[2] = 2 * s[0] * s[1] / (s[0] + s[1]);
        }
        return scores;
    }

    public static Set<String> combineNER(String[] nerArray, NERTagSet tagSet)
    {
        Set<String> result = new LinkedHashSet<String>();
        int begin = 0;
        String prePos = NERTagSet.posOf(nerArray[0]);
        for (int i = 1; i < nerArray.length; i++)
        {
            if (nerArray[i].charAt(0) == tagSet.B_TAG_CHAR || nerArray[i].charAt(0) == tagSet.S_TAG_CHAR || nerArray[i].charAt(0) == tagSet.O_TAG_CHAR)
            {
                if (i - begin > 1)
                    result.add(String.format("%d\t%d\t%s", begin, i, prePos));
                begin = i;
            }
            prePos = NERTagSet.posOf(nerArray[i]);
        }
        if (nerArray.length - 1 - begin > 1)
        {
            result.add(String.format("%d\t%d\t%s", begin, nerArray.length, prePos));
        }
        return result;
    }

    public static String[][] reshapeNER(List<String[]> ner)
    {
        String[] wordArray = new String[ner.size()];
        String[] posArray = new String[ner.size()];
        String[] nerArray = new String[ner.size()];
        reshapeNER(ner, wordArray, posArray, nerArray);
        return new String[][]{wordArray, posArray, nerArray};
    }

    public static void reshapeNER(List<String[]> collector, String[] wordArray, String[] posArray, String[] tagArray)
    {
        int i = 0;
        for (String[] tuple : collector)
        {
            wordArray[i] = tuple[0];
            posArray[i] = tuple[1];
            tagArray[i] = tuple[2];
            ++i;
        }
    }

    public static void printNERScore(Map<String, double[]> scores)
    {
        System.out.printf("%4s\t%6s\t%6s\t%6s\n", "NER", "P", "R", "F1");
        for (Map.Entry<String, double[]> entry : scores.entrySet())
        {
            String type = entry.getKey();
            double[] s = entry.getValue();
            System.out.printf("%4s\t%6.2f\t%6.2f\t%6.2f\n", type, s[0], s[1], s[2]);
        }
    }
}
