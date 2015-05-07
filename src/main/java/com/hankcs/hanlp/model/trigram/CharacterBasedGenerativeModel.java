/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/6 19:48</create-date>
 *
 * <copyright file="CharacterBasedGenerativeModel.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.hanlp.model.trigram;

import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;
import com.hankcs.hanlp.model.trigram.frequency.Probability;

import java.io.DataOutputStream;
import java.util.LinkedList;
import java.util.List;

/**
 * 基于字符的生成模型（其实就是一个TriGram文法模型，或称2阶隐马模型）
 *
 * @author hankcs
 */
public class CharacterBasedGenerativeModel implements ICacheAble
{
    double l1, l2, l3;
    Probability tf;
    static final char[] id2tag = new char[]{'b', 'm', 'e', 's'};
    static final char[] bos = {'\b', 'x'};
    static final double inf = -1e10;

    public CharacterBasedGenerativeModel()
    {
        tf = new Probability();
    }

    public void learn(List<Word> wordList)
    {
        LinkedList<char[]> sentence = new LinkedList<char[]>();
        for (IWord iWord : wordList)
        {
            String word = iWord.getValue();
            if (word.length() == 1)
            {
                sentence.add(new char[]{word.charAt(0), 's'});
            }
            else
            {
                sentence.add(new char[]{word.charAt(0), 'b'});
                for (int i = 1; i < word.length() - 1; ++i)
                {
                    sentence.add(new char[]{word.charAt(i), 'm'});
                }
                sentence.add(new char[]{word.charAt(word.length() - 1), 'e'});
            }
        }
        // 转换完毕，开始统计
        char[][] now = new char[3][];   // 定长3的队列
        now[1] = bos;
        now[2] = bos;
        tf.add(1, bos, bos);
        tf.add(2, bos);
        for (char[] i : sentence)
        {
            System.arraycopy(now, 1, now, 0, 2);
            now[2] = i;
            tf.add(1, i);   // uni
            tf.add(1, now[1], now[2]);   // bi
            tf.add(1, now);   // tri
        }
    }

    public void train()
    {
        double tl1 = 0.0;
        double tl2 = 0.0;
        double tl3 = 0.0;
        for (String key : tf.d.keySet())
        {
            if (key.length() != 6) continue;
            char[][] now = new char[][]
                    {
                            {key.charAt(0), key.charAt(1)},
                            {key.charAt(2), key.charAt(3)},
                            {key.charAt(4), key.charAt(5)},
                    };
            double c3 = div(tf.get(now) - 1, tf.get(now[0], now[1]) - 1);
            double c2 = div(tf.get(now[1], now[2]) - 1, tf.get(now[1]) - 1);
            double c1 = div(tf.get(now[2]) - 1, tf.getsum() - 1);
            if (c3 >= c1 && c3 >= c2)
                tl3 += tf.get(now);
            else if (c2 >= c1 && c2 >= c3)
                tl2 += tf.get(now);
            else if (c1 >= c2 && c1 >= c3)
                tl1 += tf.get(now);
        }

        l1 = div(tl1, tl1 + tl2 + tl3);
        l2 = div(tl2, tl1 + tl2 + tl3);
        l3 = div(tl3, tl1 + tl2 + tl3);
    }

    double log_prob(char[] s1, char[] s2, char[] s3)
    {
//        System.out.print(s1);
//        System.out.print(s2);
//        System.out.print(s3);

        double uni = l1 * tf.freq(s3);
        double bi = div(l2 * tf.get(s2, s3), tf.get(s2));
        double tri = div(l3 * tf.get(s1, s2, s3),
                         tf.get(s1, s2));
        if (uni + bi + tri == 0)
        {
//            System.out.println("inf");
            return inf;
        }
//        System.out.printf("%.2f\n", Math.log(uni + bi + tri));
        return Math.log(uni + bi + tri);
    }

    static class Path
    {
        double score;
        List<Character> path;

        public Path()
        {
            path = new LinkedList<Character>();
        }

        public void add(char tag, double score)
        {
            path.add(tag);
            this.score = score;
        }

        @Override
        public String toString()
        {
            return path + "=" + (score > inf ? score : "inf");
        }
    }

    static class Debug
    {
        Path[][] pathArray;

        public Debug(Path[][] pathArray)
        {
            this.pathArray = pathArray;
        }

        @Override
        public String toString()
        {
            StringBuilder sbInfo = new StringBuilder();
            for (Path[] line : pathArray)
            {
                for (Path path : line)
                {
                    sbInfo.append(path);
                    sbInfo.append('\n');
                }
            }

            return sbInfo.toString();
        }
    }

    public char[] tag(char[] charArray)
    {
        if (charArray.length == 0) return new char[0];
        if (charArray.length == 1) return new char[]{'s'};
        char[] tag = new char[charArray.length];
        double[][] now = new double[4][4];
        double[] first = new double[4];

        int[][][] link = new int[charArray.length][4][4];
        for (int s = 0; s < 4; ++s)
        {
            double p = (s == 1 || s == 2) ? inf : log_prob(bos, bos, new char[]{charArray[0], id2tag[s]});
            first[s] = p;
        }

        for (int f = 0; f < 4; ++f)
        {
            for (int s = 0; s < 4; ++s)
            {
                double p = first[f] + log_prob(bos, new char[]{charArray[0], id2tag[f]}, new char[]{charArray[1], id2tag[s]});
                now[f][s] = p;
                link[1][f][s] = f;
            }
        }
        for (int i = 2; i < charArray.length; i++)
        {
            double[][] pre = now;
            now = new double[4][4];
            for (int _ = 0; _ < now.length; _++)
            {
                for (int __ = 0; __ < now[_].length; __++)
                {
                    now[_][__] = -1e20;
                }
            }
            for (int f = 0; f < 4; ++f)
            {
                for (int s = 0; s < 4; ++s)
                {
                    for (int t = 0; t < 4; ++t)
                    {
                        double p = pre[f][s] + log_prob(new char[]{charArray[i - 2], id2tag[f]},
                                                        new char[]{charArray[i - 1], id2tag[s]},
                                                        new char[]{charArray[i], id2tag[t]});
                        if (p > now[s][t])
                        {
                            now[s][t] = p;
                            link[i][s][t] = f;
                        }
                    }
                }
            }
        }
        double score = inf;
        int s = 0;
        int t = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (now[i][j] > score)
                {
                    score = now[i][j];
                    s = i;
                    t = j;
                }
            }
        }
        for (int i = link.length - 1; i >= 0; --i)
        {
            tag[i] = id2tag[t];
            int f = link[i][s][t];
            t = s;
            s = f;
        }
        return tag;
    }


    private static double div(int v1, int v2)
    {
        if (v2 == 0) return 0.0;
        return v1 / (double) v2;
    }

    private static double div(double v1, double v2)
    {
        if (v2 == 0) return 0.0;
        return v1 / (double) v2;
    }

    @Override
    public void save(DataOutputStream out) throws Exception
    {
        out.writeDouble(l1);
        out.writeDouble(l2);
        out.writeDouble(l3);
        tf.save(out);
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        l1 = byteArray.nextDouble();
        l2 = byteArray.nextDouble();
        l3 = byteArray.nextDouble();
        tf.load(byteArray);
        return true;
    }
}
