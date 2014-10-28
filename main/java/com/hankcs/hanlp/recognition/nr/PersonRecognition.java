/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/26 13:52</create-date>
 *
 * <copyright file="UnknowWord.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.recognition.nr;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algoritm.Viterbi;
import com.hankcs.hanlp.algoritm.ViterbiEx;
import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.nr.PersonDictionary;
import com.hankcs.hanlp.seg.NShort.Path.Vertex;
import com.hankcs.hanlp.seg.NShort.Path.WordNet;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * 人名识别
 * @author hankcs
 */
public class PersonRecognition
{
//    static Logger logger = LoggerFactory.getLogger(PersonRecognition.class);
    public static boolean Recognition(List<Vertex> pWordSegResult, WordNet graphOptimum)
    {
        List<EnumItem<NR>> roleTagList = roleTag(pWordSegResult);
        if (HanLP.Config.DEBUG)
        {
            StringBuilder sbLog = new StringBuilder();
            Iterator<Vertex> iterator = pWordSegResult.iterator();
            for (EnumItem<NR> nrEnumItem : roleTagList)
            {
                sbLog.append('[');
                sbLog.append(iterator.next().realWord);
                sbLog.append(' ');
                sbLog.append(nrEnumItem);
                sbLog.append(']');
            }
            System.out.printf("人名角色观察：%s\n", sbLog.toString());
        }
        List<NR> nrList = viterbiExCompute(roleTagList);
        if (HanLP.Config.DEBUG)
        {
            StringBuilder sbLog = new StringBuilder();
            Iterator<Vertex> iterator = pWordSegResult.iterator();
            sbLog.append('[');
            for (NR nr : nrList)
            {
                sbLog.append(iterator.next().realWord);
                sbLog.append('/');
                sbLog.append(nr);
                sbLog.append(" ,");
            }
            if (sbLog.length() > 1) sbLog.delete(sbLog.length() - 2, sbLog.length());
            sbLog.append(']');
            System.out.printf("人名角色标注：%s\n", sbLog.toString());
        }

        PersonDictionary.parsePattern(nrList, pWordSegResult, graphOptimum);
        return true;
    }

    public static List<EnumItem<NR>> roleTag(List<Vertex> pWordSegResult)
    {
        List<EnumItem<NR>> tagList = new LinkedList<>();
        for (Vertex vertex : pWordSegResult)
        {
            // 有些双名实际上可以构成更长的三名
            if (Nature.nr == vertex.getNature() && vertex.getAttribute().totalFrequency <= 1000)
            {
                if (vertex.realWord.length() == 2)
                {
                    tagList.add(new EnumItem<NR>(NR.X, 1000));
                    continue;
                }
            }
            EnumItem<NR> nrEnumItem = PersonDictionary.dictionary.get(vertex.realWord);
            if (nrEnumItem == null)
            {
                nrEnumItem = new EnumItem<>(NR.A, PersonDictionary.transformMatrixDictionary.getTotalFrequency(NR.A));
            }
            tagList.add(nrEnumItem);
        }
        return tagList;
    }

    /**
     * 维特比算法求解最优标签
     * @param roleTagList
     * @return
     */
    public static List<NR> viterbiCompute(List<EnumItem<NR>> roleTagList)
    {
        List<NR> resultList = new LinkedList<>();
        // HMM五元组
        int[] observations = new int[roleTagList.size()];
        for (int i = 0; i < observations.length; ++i)
        {
            observations[i] = i;
        }
        double[][] emission_probability = new double[PersonDictionary.transformMatrixDictionary.ordinaryMax][observations.length];
        for (int i = 0; i < emission_probability.length; ++i)
        {
            for (int j = 0; j < emission_probability[i].length; ++j)
            {
                emission_probability[i][j] = 1e8;
            }
        }
        for (int s : PersonDictionary.transformMatrixDictionary.states)
        {
            Iterator<EnumItem<NR>> iterator = roleTagList.iterator();
            for (int o : observations)
            {
                NR sNR = NR.values()[s];
                EnumItem<NR> item = iterator.next();
                double frequency = item.getFrequency(sNR);
                if (frequency == 0)
                {
                    emission_probability[s][o] = 1e8;
                }
                else
                {
                    emission_probability[s][o] = -Math.log(frequency / PersonDictionary.transformMatrixDictionary.getTotalFrequency(sNR));
                }

            }
        }
        int[] result = Viterbi.compute(observations,
                                        PersonDictionary.transformMatrixDictionary.states,
                                        PersonDictionary.transformMatrixDictionary.start_probability,
                                        PersonDictionary.transformMatrixDictionary.transititon_probability,
                                        emission_probability
        );
        for (int r : result)
        {
            resultList.add(NR.values()[r]);
        }
        return resultList;
    }

    /**
     * 维特比算法求解最优标签
     * @param roleTagList
     * @return
     */
    public static List<NR> viterbiExCompute(List<EnumItem<NR>> roleTagList)
    {
        ViterbiEx<NR> viterbiEx = new ViterbiEx<>(roleTagList, PersonDictionary.transformMatrixDictionary);
        return viterbiEx.computeTagList();
    }
}
