/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/8 0:52</create-date>
 *
 * <copyright file="MutualInformationPhraseExactor.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.phrase;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.occurrence.Occurrence;
import com.hankcs.hanlp.corpus.occurrence.PairFrequency;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.dictionary.stopword.Filter;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NotionalTokenizer;

import java.util.LinkedList;
import java.util.List;

import static com.hankcs.hanlp.corpus.tag.Nature.nx;
import static com.hankcs.hanlp.corpus.tag.Nature.t;

/**
 * 利用互信息和左右熵的短语提取器
 * @author hankcs
 */
public class MutualInformationEntropyPhraseExtractor implements IPhraseExtractor
{
    @Override
    public List<String> extractPhrase(String text, int size)
    {
        List<String> phraseList = new LinkedList<String>();
        Occurrence occurrence = new Occurrence();
        Filter[] filterChain = new Filter[]
                {
                        CoreStopWordDictionary.FILTER,
                        new Filter()
                        {
                            @Override
                            public boolean shouldInclude(Term term)
                            {
                                if (term.nature == t || term.nature == nx)
                                    return false;
                                return true;
                            }
                        }
                };
        for (List<Term> sentence : NotionalTokenizer.seg2sentence(text, filterChain))
        {
            if (HanLP.Config.DEBUG)
            {
                System.out.println(sentence);
            }
            occurrence.addAll(sentence);
        }
        occurrence.compute();
        if (HanLP.Config.DEBUG)
        {
            System.out.println(occurrence);
            for (PairFrequency phrase : occurrence.getPhraseByMi())
            {
                System.out.print(phrase.getKey().replace(Occurrence.RIGHT, '→') + "\tmi=" + phrase.mi + " , ") ;
            }
            System.out.println();
            for (PairFrequency phrase : occurrence.getPhraseByLe())
            {
                System.out.print(phrase.getKey().replace(Occurrence.RIGHT, '→') + "\tle=" + phrase.le + " , ");
            }
            System.out.println();
            for (PairFrequency phrase : occurrence.getPhraseByRe())
            {
                System.out.print(phrase.getKey().replace(Occurrence.RIGHT, '→') + "\tre=" + phrase.re + " , ");
            }
            System.out.println();
            for (PairFrequency phrase : occurrence.getPhraseByScore())
            {
                System.out.print(phrase.getKey().replace(Occurrence.RIGHT, '→') + "\tscore=" + phrase.score + " , ");
            }
            System.out.println();
        }

        for (PairFrequency phrase : occurrence.getPhraseByScore())
        {
            if (phraseList.size() == size) break;
            phraseList.add(phrase.first + phrase.second);
        }
        return phraseList;
    }

    /**
     * 一句话提取
     * @param text
     * @param size
     * @return
     */
    public static List<String> extract(String text, int size)
    {
        IPhraseExtractor extractor = new MutualInformationEntropyPhraseExtractor();
        return extractor.extractPhrase(text, size);
    }

//    public static void main(String[] args)
//    {
//        MutualInformationEntropyPhraseExtractor extractor = new MutualInformationEntropyPhraseExtractor();
//        String text = "算法工程师\n" +
//                "算法（Algorithm）是一系列解决问题的清晰指令，也就是说，能够对一定规范的输入，在有限时间内获得所要求的输出。如果一个算法有缺陷，或不适合于某个问题，执行这个算法将不会解决这个问题。不同的算法可能用不同的时间、空间或效率来完成同样的任务。一个算法的优劣可以用空间复杂度与时间复杂度来衡量。算法工程师就是利用算法处理事物的人。\n" +
//                "\n" +
//                "1职位简介\n" +
//                "算法工程师是一个非常高端的职位；\n" +
//                "专业要求：计算机、电子、通信、数学等相关专业；\n" +
//                "学历要求：本科及其以上的学历，大多数是硕士学历及其以上；\n" +
//                "语言要求：英语要求是熟练，基本上能阅读国外专业书刊；\n" +
//                "必须掌握计算机相关知识，熟练使用仿真工具MATLAB等，必须会一门编程语言。\n" +
//                "\n" +
//                "2研究方向\n" +
//                "视频算法工程师、图像处理算法工程师、音频算法工程师 通信基带算法工程师\n" +
//                "\n" +
//                "3目前国内外状况\n" +
//                "目前国内从事算法研究的工程师不少，但是高级算法工程师却很少，是一个非常紧缺的专业工程师。算法工程师根据研究领域来分主要有音频/视频算法处理、图像技术方面的二维信息算法处理和通信物理层、雷达信号处理、生物医学信号处理等领域的一维信息算法处理。\n" +
//                "在计算机音视频和图形图像技术等二维信息算法处理方面目前比较先进的视频处理算法：机器视觉成为此类算法研究的核心；另外还有2D转3D算法(2D-to-3D conversion)，去隔行算法(de-interlacing)，运动估计运动补偿算法(Motion estimation/Motion Compensation)，去噪算法(Noise Reduction)，缩放算法(scaling)，锐化处理算法(Sharpness)，超分辨率算法(Super Resolution),手势识别(gesture recognition),人脸识别(face recognition)。\n" +
//                "在通信物理层等一维信息领域目前常用的算法：无线领域的RRM、RTT，传送领域的调制解调、信道均衡、信号检测、网络优化、信号分解等。\n" +
//                "另外数据挖掘、互联网搜索算法也成为当今的热门方向。\n" +
//                "算法工程师逐渐往人工智能方向发展。";
////        System.out.println(text);
//        List<String> phraseList = extractor.extractPhrase(text, 10);
//        System.out.println(phraseList);
//    }
}
