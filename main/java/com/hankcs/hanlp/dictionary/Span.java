/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/28 21:54</create-date>
 *
 * <copyright file="Span.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.seg.NShort.Path.WordResult;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.Utility;

/**
 * @author hankcs
 */
public class Span
{
    //The number of unknown word
    public int m_nUnknownWordsCount;
    //The start and ending possition of unknown position
    public int[][] m_nUnknownWords;
    //The possibility of unknown words
    public double[] m_dWordsPossibility;
    public ContextStat m_context;

    private TAG_TYPE m_tagType; //The type of tagging
    private int m_nStartPos;

    private int[] m_nBestTag = new int[Predefine.MAX_WORDS_PER_SENTENCE];
    //Record the Best Tag

    private String[] m_sWords;
    private int[] m_nWordPosition;
    private int[][] m_nTags;
    private int[][] m_nBestPrev;
    private double[][] m_dFrequency;
    private int m_nCurLength;

    public Span(ContextStat contextStat, TAG_TYPE tagType)
    {
        m_nUnknownWords = new int[Predefine.MAX_UNKNOWN_PER_SENTENCE][2];
        m_dWordsPossibility = new double[Predefine.MAX_UNKNOWN_PER_SENTENCE];
        m_sWords = new String[Predefine.MAX_WORDS_PER_SENTENCE];
        m_nWordPosition = new int[Predefine.MAX_WORDS_PER_SENTENCE];
        m_nTags = new int[Predefine.MAX_WORDS_PER_SENTENCE][Predefine.MAX_POS_PER_WORD];
        m_nBestPrev = new int[Predefine.MAX_WORDS_PER_SENTENCE][Predefine.MAX_POS_PER_WORD];
        m_dFrequency = new double[Predefine.MAX_WORDS_PER_SENTENCE][Predefine.MAX_POS_PER_WORD];
        if (m_tagType != TAG_TYPE.TT_NORMAL)
            m_nTags[0][0] = 100;
            //Begin tag
        else
            m_nTags[0][0] = 0;
        //Begin tag

        m_nTags[0][1] = -1;
        m_dFrequency[0][0] = 0;
        m_nCurLength = 1;
        m_nUnknownWordsCount = 0;
        m_nStartPos = 0;
        m_nWordPosition[1] = 0;
        m_sWords[0] = null;

        m_tagType = tagType;
        m_context = contextStat;
    }

    private boolean Disamb()
    {
        int i, j, k, nMinCandidate;
        double dMinFee = 0, dTmp;
        for (i = 1; i < m_nCurLength; i++)
        //For every word
        {
            for (j = 0; m_nTags[i][j] >= 0; j++)
            //For every word
            {
                nMinCandidate = Predefine.MAX_POS_PER_WORD + 1;
                for (k = 0; m_nTags[i - 1][k] >= 0; k++)
                {
                    dTmp = -Math.log(m_context.GetContextPossibility(0, m_nTags[i - 1][k], m_nTags[i][j]));
                    dTmp += m_dFrequency[i - 1][k]; //Add the fees
                    if (nMinCandidate > 10 || dTmp < dMinFee)
                    //Get the minimum fee
                    {
                        nMinCandidate = k;
                        dMinFee = dTmp;
                    }
                }

                m_nBestPrev[i][j] = nMinCandidate; //The best previous for j
                m_dFrequency[i][j] = m_dFrequency[i][j] + dMinFee;
            }
        }
        return true;
    }

    private boolean Reset()
    {
        return Reset(true);
    }

    private boolean Reset(boolean bContinue)
    {
        if (!bContinue)
        {
            //||CC_Find("。！”〕〉》」〗】",m_sWords[m_nCurLength-1])
            if (m_tagType != TAG_TYPE.TT_NORMAL)
                //Get the last POS in the last sentence
                m_nTags[0][0] = 100;
                //Begin tag
            else
                m_nTags[0][0] = 0;
            //Begin tag
            m_nUnknownWordsCount = 0;
            m_dFrequency[0][0] = 0;
            m_nStartPos = 0;
        } else
        {
            m_nTags[0][0] = m_nTags[m_nCurLength - 1][0];
            //Get the last POS in the last sentence
            m_dFrequency[0][0] = m_dFrequency[m_nCurLength - 1][0];
        }
        m_nTags[0][1] = -1;
        //Get the last POS in the last sentence,set the -1 as end flag
        m_nCurLength = 1;
        m_nWordPosition[1] = m_nStartPos;
        m_sWords[0] = null;
        return true;
    }

    public boolean LoadContext(String sFilename)
    {
        return m_context.Load(sFilename);
    }

    private boolean GetBestPOS()
    {
        Disamb();
        for (int i = m_nCurLength - 1, j = 0; i > 0; i--)
        //,j>=0
        {
            if (m_sWords[i] != null)
            {
                //Not virtual ending
                m_nBestTag[i] = m_nTags[i][j]; //Record the best POS and its possibility
            }
            j = m_nBestPrev[i][j];
        }
        int nEnd = m_nCurLength; //Set the end of POS tagging
        if (m_sWords[m_nCurLength - 1] == null)
            nEnd = m_nCurLength - 1;
        m_nBestTag[nEnd] = -1;
        return true;
    }

    public boolean PersonRecognize()
    {
        StringBuilder sb = new StringBuilder();

        int i;
        String sPOS = "z", sPersonName;
        String[] sPatterns = {"BBCD", "BBC", "BBE", "BBZ", "BCD", "BEE", "BE", "BG", "BXD", "BZ", "CDCD", "CD", "EE", "FB", "Y", "XD", ""};
        // 下面这些概率的和为1
        double[] dFactor = {0.003606, 0.000021, 0.001314, 0.000315, 0.656624, 0.000021, 0.146116, 0.009136, 0.000042, 0.038971, 0, 0.090367, 0.000273, 0.009157, 0.034324, 0.009735, 0};

         /*------------------------------------
         About parameter:

         BBCD  343      0.003606
         BBC   2        0.000021
         BBE   125      0.001314
         BBZ   30       0.000315
         BCD   62460    0.656624
         BEE   0        0.000000
         BE    13899    0.146116
         BG    869      0.009136
         BXD   4        0.000042
         BZ    3707     0.038971
         CD    8596     0.090367
         EE    26       0.000273
         FB    871      0.009157
         Y     3265     0.034324
         XD    926      0.009735

         The person recognition patterns set
         BBCD:姓+姓+名1+名2;
         BBE: 姓+姓+单名;
         BBZ: 姓+姓+双名成词;
         BCD: 姓+名1+名2;
         BE:  姓+单名;
         BEE: 姓+单名+单名;韩磊磊
         BG:  姓+后缀
         BXD: 姓+姓双名首字成词+双名末字
         BZ:  姓+双名成词;
         B:   姓
         CD:  名1+名2;
         EE:  单名+单名;
         FB:  前缀+姓
         XD:  姓双名首字成词+双名末字
         Y:   姓单名成词
         ------------------------------------*/

        int[] nPatternLen = {4, 3, 3, 3, 3, 3, 2, 2, 3, 2, 4, 2, 2, 2, 1, 2, 0};

        //Convert to string from POS
        sb.append('z');
        //
        for (i = 1; m_nBestTag[i] > -1; i++)
            sb.append((char) (m_nBestTag[i] + 'A'));

        sPOS = sb.toString();

        int j = 1, k, nPos; //Find the proper pattern from the first POS
        int nLittleFreqCount; //Counter for the person name role with little frequecy
        boolean bMatched = false;
        while (j < i)
        {
            bMatched = false;
            for (k = 0; !bMatched && nPatternLen[k] > 0; k++)
            {
                if (sPatterns[k].equals(sPOS.substring(j)) &&
                        !m_sWords[j - 1].equals("·") && !m_sWords[j + nPatternLen[k]].equals("·"))
                {
                    //Find the proper pattern k
                    if (sPatterns[k].equals("FB") && (sPOS.charAt(j + 2) == 'E' || sPOS.charAt(j + 2) == 'C' || sPOS.charAt(j + 2) == 'G'))
                    {
                        //Rule 1 for exclusion:前缀+姓+名1(名2): 规则(前缀+姓)失效；
                        continue;
                    }

                  /*
                  if((strcmp(sPatterns[k],"BEE")==0||strcmp(sPatterns[k],"EE")==0)&&strcmp(m_sWords[j+nPatternLen[k]-1],m_sWords[j+nPatternLen[k]-2])!=0)
                  {//Rule 2 for exclusion:姓+单名+单名:单名+单名 若EE对应的字不同，规则失效.如：韩磊磊
                  continue;
                  }

                  if(strcmp(sPatterns[k],"B")==0&&m_nBestTag[j+1]!=12)
                  {//Rule 3 for exclusion: 若姓后不是后缀，规则失效.如：江主席、刘大娘
                  continue;
                  }
                   */
                    //Get the possible name

                    nPos = j; //Record the person position in the tag sequence
                    sPersonName = null;
                    nLittleFreqCount = 0; //Record the number of role with little frequency
                    while (nPos < j + nPatternLen[k])
                    {
                        //Get the possible person name
                        //
                        if (m_nBestTag[nPos] < 4 && ZZPersonDictionary.getFrequency(m_sWords[nPos], m_nBestTag[nPos]) < Predefine.LITTLE_FREQUENCY)
                            nLittleFreqCount++;
                        //The counter increase
                        sPersonName += m_sWords[nPos];
                        nPos += 1;
                    }
                  /*
                  if(IsAllForeign(sPersonName)&&personDict.GetFrequency(m_sWords[j],1)<LITTLE_FREQUENCY)
                  {//Exclusion foreign name
                  //Rule 2 for exclusion:若均为外国人名用字 规则(名1+名2)失效
                  j+=nPatternLen[k]-1;
                  continue;
                  }
                   */
                    if (sPatterns[k].equals("CDCD"))
                    {
                        //Rule for exclusion
                        //规则(名1+名2+名1+名2)本身是排除规则:女高音歌唱家迪里拜尔演唱
                        //Rule 3 for exclusion:含外国人名用字 规则适用
                        //否则，排除规则失效:黑妞白妞姐俩拔了头筹。
                        if (Utility.getForeignCharCount(sPersonName) > 0)
                            j += nPatternLen[k] - 1;
                        continue;
                    }
                  /*
                  if(strcmp(sPatterns[k],"CD")==0&&IsAllForeign(sPersonName))
                  {//
                  j+=nPatternLen[k]-1;
                  continue;
                  }
                  if(nLittleFreqCount==nPatternLen[k]||nLittleFreqCount==3)
                  //马哈蒂尔;小扎耶德与他的中国阿姨胡彩玲受华黎明大使之邀，
                  //The all roles appear with two lower frequecy,we will ignore them
                  continue;
                   */
                    m_nUnknownWords[m_nUnknownWordsCount][0] = m_nWordPosition[j];
                    m_nUnknownWords[m_nUnknownWordsCount][1] = m_nWordPosition[j + nPatternLen[k]];
                    m_dWordsPossibility[m_nUnknownWordsCount] = -Math.log(dFactor[k]) + ComputePossibility(j, nPatternLen[k], ZZPersonDictionary.getUnknownWordDictionary());
                    //Mutiply the factor
                    m_nUnknownWordsCount += 1;
                    j += nPatternLen[k];
                    bMatched = true;
                }
            }
            if (!bMatched)
                //Not matched, add j by 1
                j += 1;
        }
        return true;
    }

    private int GuessPOS(int nIndex)
    {
        int pSubIndex;
        int j = 0, i = nIndex, nCharType;
        int nLen;
        switch (m_tagType)
        {
            case TT_NORMAL:
                m_nTags[i][j] = Utility.GetPOSValue("x"); //对于没有任何词性的词认为是字符串
                m_dFrequency[i][j++] = 0;
                break;
            case TT_PERSON:
                j = 0;
                if ("××".contains(m_sWords[nIndex]))
                {
                    m_nTags[i][j] = 6;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 6) + 1);
                } else
                {
                    m_nTags[i][j] = 0;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 0) + 1);
                    nLen = m_sWords[nIndex].length();
                    if (nLen >= 2)
                    {
                        m_nTags[i][j] = 0;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 0) + 1);
                        m_nTags[i][j] = 11;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 11) * 8);
                        m_nTags[i][j] = 12;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 12) * 8);
                        m_nTags[i][j] = 13;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 13) * 8);
                    } else if (nLen == 1)
                    {
                        m_nTags[i][j] = 0;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 0) + 1);
                        nCharType = Utility.charType(m_sWords[nIndex].toCharArray()[0]);
                        if (nCharType == Predefine.CT_OTHER || nCharType == Predefine.CT_CHINESE)
                        {
                            m_nTags[i][j] = 1;
                            m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 1) + 1);
                            m_nTags[i][j] = 2;
                            m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 2) + 1);
                            m_nTags[i][j] = 3;
                            m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 3) + 1);
                            m_nTags[i][j] = 4;
                            m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 4) + 1);
                        }
                        m_nTags[i][j] = 11;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 11) * 8);
                        m_nTags[i][j] = 12;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 12) * 8);
                        m_nTags[i][j] = 13;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 13) * 8);
                    }
                }
                break;
            case TT_PLACE:
                j = 0;
                m_nTags[i][j] = 0;
                m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 0) + 1);
                nLen = m_sWords[nIndex].length();
                if (nLen >= 2)
                {
                    m_nTags[i][j] = 11;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 11) * 8);
                    m_nTags[i][j] = 12;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 12) * 8);
                    m_nTags[i][j] = 13;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 13) * 8);
                } else if (nLen == 1)
                {
                    m_nTags[i][j] = 0;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 0) + 1);
                    nCharType = Utility.charType(m_sWords[nIndex].toCharArray()[0]);
                    if (nCharType == Predefine.CT_OTHER || nCharType == Predefine.CT_CHINESE)
                    {
                        m_nTags[i][j] = 1;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 1) + 1);
                        m_nTags[i][j] = 2;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 2) + 1);
                        m_nTags[i][j] = 3;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 3) + 1);
                        m_nTags[i][j] = 4;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 4) + 1);
                    }
                    m_nTags[i][j] = 11;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 11) * 8);
                    m_nTags[i][j] = 12;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 12) * 8);
                    m_nTags[i][j] = 13;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 13) * 8);
                }
                break;
            case TT_TRANS_PERSON:
                j = 0;
                nLen = m_sWords[nIndex].length();

                m_nTags[i][j] = 0;
                m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 0) + 1);

                if (!Utility.isAllChinese(m_sWords[nIndex]))
                {
                    if (Utility.isAllLetter(m_sWords[nIndex]))
                    {
                        m_nTags[i][j] = 1;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 1) + 1);
                        m_nTags[i][j] = 11;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 11) + 1);
                        m_nTags[i][j] = 2;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 2) * 2 + 1);
                        m_nTags[i][j] = 3;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 3) * 2 + 1);
                        m_nTags[i][j] = 12;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 12) * 2 + 1);
                        m_nTags[i][j] = 13;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 13) * 2 + 1);
                    }
                    m_nTags[i][j] = 41;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 41) * 8);
                    m_nTags[i][j] = 42;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 42) * 8);
                    m_nTags[i][j] = 43;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 43) * 8);
                } else if (nLen >= 2)
                {
                    m_nTags[i][j] = 41;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 41) * 8);
                    m_nTags[i][j] = 42;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 42) * 8);
                    m_nTags[i][j] = 43;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 43) * 8);
                } else if (nLen == 1)
                {
                    nCharType = Utility.charType(m_sWords[nIndex].toCharArray()[0]);
                    if (nCharType == Predefine.CT_OTHER || nCharType == Predefine.CT_CHINESE)
                    {
                        m_nTags[i][j] = 1;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 1) * 2 + 1);
                        m_nTags[i][j] = 2;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 2) * 2 + 1);
                        m_nTags[i][j] = 3;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 3) * 2 + 1);
                        m_nTags[i][j] = 30;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 30) * 8 + 1);
                        m_nTags[i][j] = 11;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 11) * 4 + 1);
                        m_nTags[i][j] = 12;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 12) * 4 + 1);
                        m_nTags[i][j] = 13;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 13) * 4 + 1);
                        m_nTags[i][j] = 21;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 21) * 2 + 1);
                        m_nTags[i][j] = 22;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 22) * 2 + 1);
                        m_nTags[i][j] = 23;
                        m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 23) * 2 + 1);
                    }
                    m_nTags[i][j] = 41;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 41) * 8);
                    m_nTags[i][j] = 42;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 42) * 8);
                    m_nTags[i][j] = 43;
                    m_dFrequency[i][j++] = 1.0 / (m_context.GetFrequency(0, 43) * 8);
                }
                break;
            default:
                break;
        }
        pSubIndex = j;
        return pSubIndex;
    }

    private double ComputePossibility(int nStartPos, int nLength, UnknownWordDictionary dict)
    {
        double dRetValue = 0, dPOSPoss;
        //dPOSPoss: the possibility of a POS appears
        //dContextPoss: The possibility of context POS appears
        int nFreq;
        for (int i = nStartPos; i < nStartPos + nLength; i++)
        {
            nFreq = dict.getFrequency(m_sWords[i], m_nBestTag[i]);
            //nFreq is word being the POS
            dPOSPoss = Math.log((double)(m_context.GetFrequency(0, m_nBestTag[i]) + 1)) - Math.log((double)(nFreq + 1));
            dRetValue += dPOSPoss;
            /*
             if(i<nStartPos+nLength-1)
             {
                dContextPoss=log((double)(m_context.GetContextPossibility(0,m_nBestTag[i],m_nBestTag[i+1])+1));
                dRetValue+=dPOSPoss-dContextPoss;
             }
             */
        }
        return dRetValue;
    }

    private int GetFrom(WordResult[] pWordItems, int nIndex, UnknownWordDictionary dictUnknown)
    {
        int[] aPOS = new int[Predefine.MAX_POS_PER_WORD];
        int[] aFreq = new int[Predefine.MAX_POS_PER_WORD];
        int nFreq = 0, j = 0, nRetPos = 0, nWordsIndex = 0;
        boolean bSplit = false; //Need to split in Transliteration recognition
        int i = 1, nPOSCount;
        String sCurWord; //Current word

        nWordsIndex = i + nIndex - 1;
        for (i = 1; i < Predefine.MAX_WORDS_PER_SENTENCE && nWordsIndex < pWordItems.length; ++i)
        {
            if (m_tagType == TAG_TYPE.TT_NORMAL || !dictUnknown.IsExist(pWordItems[nWordsIndex].sWord, 44))
            {
                m_sWords[i] = pWordItems[nWordsIndex].sWord; //store current word
                m_nWordPosition[i + 1] = m_nWordPosition[i] + m_sWords[i].length();
            }
            else
            {
                if (!bSplit)
                {
                    m_sWords[i] = pWordItems[nWordsIndex].sWord.substring(0, 1);
                    //store current word
                    bSplit = true;
                }
                else
                {
                    m_sWords[i] = pWordItems[nWordsIndex].sWord.substring(1);
                    //store current word
                    bSplit = false;
                }
                m_nWordPosition[i + 1] = m_nWordPosition[i] + m_sWords[i].length();
            }
            //Record the position of current word
            m_nStartPos = m_nWordPosition[i + 1];
            //Move the Start POS to the ending
            if (m_tagType != TAG_TYPE.TT_NORMAL)
            {
                //Get the POSs from the unknown recognition dictionary
                sCurWord = m_sWords[i];
                if (m_tagType == TAG_TYPE.TT_TRANS_PERSON && i > 0 && m_sWords[i - 1] != null &&
                        Utility.charType(m_sWords[i - 1].toCharArray()[0]) == Predefine.CT_CHINESE)
                {
                    if (m_sWords[i] == ".")
                        sCurWord = "．";
                    else if (m_sWords[i] == "-")
                        sCurWord = "－";
                }

                {
                    UnknownWordDictionary.Attribute info = dictUnknown.GetWordInfo(sCurWord);
                    if (info != null)
                    {
                        nPOSCount = info.pos.length + 1;
                        for (j = 0; j < info.pos.length; j++)
                        {
                            //Get the POS set of sCurWord in the unknown dictionary
                            m_nTags[i][ j] = info.pos[j];
                            m_dFrequency[i][ j] = -Math.log((double)(1 + info.frequency[j])) +
                                    Math.log((double)(m_context.GetFrequency(0, info.pos[j]) + nPOSCount));
                        }
                    }
                    else
                    {
                        nPOSCount = 1;
                        j = 0;
                    }
                }

                //Get the POS set of sCurWord in the core dictionary
                //We ignore the POS in the core dictionary and recognize them as other (0).
                //We add their frequency to get the possibility as POS 0
                if (m_sWords[i].equals("始##始") )
                {
                    m_nTags[i][ j] = 100;
                    m_dFrequency[i][ j] = 0;
                    j++;
                }
                else if (m_sWords[i].equals("末##末"))
                {
                    m_nTags[i][ j] = 101;
                    m_dFrequency[i][ j] = 0;
                    j++;
                }
                else
                {
                    CoreDictionary.Attribute info = CoreDictionary.GetWordInfo(m_sWords[i]);
                    nFreq = 0;
                    if (info != null)
                    {
                        for (int k = 0; k < info.frequency.length; k++)
                        {
                            nFreq += info.frequency[k];
                        }
                        if (info.frequency.length > 0)
                        {
                            m_nTags[i][ j] = 0;
                            //m_dFrequency[i][j]=(double)(1+nFreq)/(double)(m_context.GetFrequency(0,0)+1);
                            m_dFrequency[i][ j] = -Math.log((double)(1 + nFreq)) + Math.log((double)(m_context.GetFrequency(0, 0) + nPOSCount));
                            j++;
                        }
                    }
                }
            }
//            else
//            //For normal POS tagging
//            {
//                j = 0;
//                //Get the POSs from the unknown recognition dictionary
//                if (pWordItems[nWordsIndex].nPOS != null)
//                {
//                    //The word has  is only one POS value
//                    //We have record its POS and nFrequncy in the items.
//                    m_nTags[i][j] = pWordItems[nWordsIndex].nPOS;
//                    m_dFrequency[i, j] = -Math.Log(pWordItems[nWordsIndex].dValue) + Math.Log((double)(m_context.GetFrequency(0, m_nTags[i, j]) + 1));
//                    if (m_dFrequency[i, j] < 0)
//                    //Not permit the value less than 0
//                    m_dFrequency[i, j] = 0;
//                    j++;
//                }
//                else
//                {
//                    //The word has multiple POSs, we should retrieve the information from Core Dictionary
//                    if (pWordItems[nWordsIndex].nPOS < 0)
//                    {
//                        //The word has  is only one POS value
//                        //We have record its POS and nFrequncy in the items.
//                        m_nTags[i, j] = -pWordItems[nWordsIndex].nPOS;
//                        m_dFrequency[i, j++] = pWordItems[nWordsIndex].dValue;
//                    }
//                    //dictCore.GetHandle(m_sWords[i], &nCount, aPOS, aFreq);
//                    info = dictCore.GetWordInfo(m_sWords[i]);
//                    if (info != null)
//                    {
//                        nPOSCount = info.Count;
//                        for (; j < info.Count; j++)
//                        {
//                            //Get the POS set of sCurWord in the unknown dictionary
//                            m_nTags[i, j] = info.POSs[j];
//                            m_dFrequency[i, j] = -Math.Log(1 + info.Frequencies[j]) + Math.Log(m_context.GetFrequency(0, m_nTags[i, j]) + nPOSCount);
//                        }
//                    }
//                }
//            }
            if (j == 0)
            {
                //We donot know the POS, so we have to guess them according lexical knowledge
                j = GuessPOS(i); //Guess the POS of current word
            }
            m_nTags[i][ j] = -1; //Set the ending POS
            if (j == 1 && m_nTags[i][ j] != Predefine.CT_SENTENCE_BEGIN)
            //No ambuguity
            {
                //No ambuguity, so we can break from the loop
                i++;
                m_sWords[i] = null;
                break;
            }
            if (!bSplit)
                nWordsIndex++;
        }
        if (nWordsIndex == pWordItems.length)
            nRetPos = -1;
        //Reaching ending

        if (m_nTags[i - 1][ 1] != -1)
        //||m_sWords[i][0]==0
        {
            //Set end for words like "张/华/平"
            if (m_tagType != TAG_TYPE.TT_NORMAL)
                m_nTags[i][ 0] = 101;
            else
            m_nTags[i][ 0] = 1;

            m_dFrequency[i][ 0] = 0;
            m_sWords[i] = null; //Set virtual ending
            m_nTags[i++][ 1] = -1;
        }
        m_nCurLength = i; //The current word count
        if (nRetPos != -1)
            return nWordsIndex + 1;
        //Next start position
        return -1; //Reaching ending
    }

    public boolean POSTagging(WordResult[] pWordItems, UnknownWordDictionary dictUnknown)
    {
        //pWordItems: Items; nItemCount: the count of items;core dictionary and unknown recognition dictionary
        int i = 0, j, nStartPos;
        Reset(false);
        while (i > -1 && i < pWordItems.length && pWordItems[i].sWord != null)
        {
            nStartPos = i; //Start Position
            i = GetFrom(pWordItems, nStartPos, dictUnknown);
            GetBestPOS();
            switch (m_tagType)
            {
//                case TAG_TYPE.TT_NORMAL:
//                    //normal POS tagging
//                    j = 1;
//                    while (m_nBestTag[j] != -1 && j < m_nCurLength)
//                    {
//                        //Store the best POS tagging
//                        pWordItems[j + nStartPos - 1].nPOS = m_nBestTag[j];
//                        //Let 。be 0
//                        if (pWordItems[j + nStartPos - 1].dValue > 0 && dictCore.IsExist(pWordItems[j + nStartPos - 1].sWord, -1))
//                            //Exist and update its frequncy as a POS value
//                            pWordItems[j + nStartPos - 1].dValue = dictCore.GetFrequency(pWordItems[j + nStartPos - 1].sWord, m_nBestTag[j]);
//                        j += 1;
//                    }
//                    break;
                case TT_PERSON:
                    //Person recognition
                    PersonRecognize();
                    break;
                case TT_PLACE:
                    //Place name recognition
                case TT_TRANS_PERSON:
                    //Transliteration Person
//                    PlaceRecognize(dictCore, dictUnknown);
                    break;
                default:
                    break;
            }
            Reset();
        }
        return true;
    }

    public enum TAG_TYPE
    {
        TT_NORMAL,
        TT_PERSON,
        TT_PLACE,
        TT_TRANS_PERSON
    }
}
