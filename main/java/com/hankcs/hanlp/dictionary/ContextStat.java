/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/28 21:18</create-date>
 *
 * <copyright file="ContextStat.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;

/**
 * 上下文类，用于词性标注等
 * @author hankcs
 */
public class ContextStat
{
    /**
     * 标注集的个数
     */
    private int m_nTableLen;
    /**
     * 标注集
     */
    private int[] m_pSymbolTable;
    /**
     * 上下文信息单向链表
     */
    private ContextItem m_pContext;

    public boolean Load(String sFilename)
    {
        boolean isSuccess = true;
        ContextItem pCur = null, pPre = null;
        BufferedReader br = null;
        try
        {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(sFilename)));

            String line = br.readLine();
            m_nTableLen = Integer.parseInt(line);
            m_pSymbolTable = new int[m_nTableLen]; //new buffer for symbol

            line = br.readLine();
            String[] param = line.split(" "); 
            for (int i = 0; i < m_nTableLen; ++i)  //write the symbol table
            {
                m_pSymbolTable[i] = Integer.parseInt(param[i]);
            }

            while ((line = br.readLine()) != null)
            {
                //Read the context 
                pCur = new ContextItem();
                pCur.next = null;
                pCur.nKey = Integer.parseInt(line);
                line = br.readLine();
                pCur.nTotalFreq = Integer.parseInt(line);

                pCur.aTagFreq = new int[m_nTableLen];
                line = br.readLine();
                param = line.split(" ");
                for (int i = 0; i < m_nTableLen; ++i)     //the every POS frequency
                {
                    pCur.aTagFreq[i] = Integer.parseInt(param[i]);
                }


                pCur.aContextArray = new int[m_nTableLen][];
                for (int i = 0; i < m_nTableLen; ++i)
                {
                    pCur.aContextArray[i] = new int[m_nTableLen];
                    line = br.readLine();
                    param = line.split(" ");
                    for (int j = 0; j < m_nTableLen; j++)
                    {
                        pCur.aContextArray[i][j] = Integer.parseInt(param[j]);
                    }
                }

                if (pPre == null)
                    m_pContext = pCur;
                else
                    pPre.next = pCur;

                pPre = pCur;
            }
            br.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();
            isSuccess = false;
        }

//        LogManager.getLogger().info("上下文词典载入结果：" + isSuccess);
        return isSuccess;
    }

    /**
     * 返回nKey为指定nKey的结点，如果没找到，则返回前一个结点
     * @param nKey
     * @return
     */
    public ContextItem GetItem(int nKey)
    {
        ContextItem pItemRet = null;
        ContextItem pCur = m_pContext, pPrev = null;
        if (nKey == 0 && m_pContext != null)
        {
            pItemRet = m_pContext;
            return pItemRet;
        }

        while (pCur != null && pCur.nKey < nKey)
        {
            //delete the context array
            pPrev = pCur;
            pCur = pCur.next;
        }

        if (pCur != null && pCur.nKey == nKey)
        {
            //find it and return the current item
            pItemRet = pCur;
            return pItemRet;
        }

        pItemRet = pPrev;
        return pItemRet;
    }

    /**
     * 搜索nSymbol出现的频次
     * @param nKey
     * @param nSymbol
     * @return
     */
    public int GetFrequency(int nKey, int nSymbol)
    {
        ContextItem pFound = GetItem(nKey);

        int nIndex, nFrequency = 0;
        if (pFound == null)
            //Not found such a item
            return 0;

        nIndex = Arrays.binarySearch(m_pSymbolTable, nSymbol);
        if (nIndex < 0)
            //error finding the symbol
            return 0;

        nFrequency = pFound.aTagFreq[nIndex]; //Add the frequency
        return nFrequency;
    }

    public int GetFrequency(int nSymbol)
    {
        return GetFrequency(0, nSymbol);
    }

    public double GetContextPossibility(int nKey, int nPrev, int nCur)
    {
        ContextItem pCur;
        int nCurIndex = Arrays.binarySearch(m_pSymbolTable, nCur);
        int nPrevIndex = Arrays.binarySearch(m_pSymbolTable, nPrev);

        //return a lower value, not 0 to prevent data sparse
        pCur = GetItem(nKey);
        if (pCur.nKey != nKey || nCurIndex <= -1 || nPrevIndex <= -1 ||
                pCur.aTagFreq[nPrevIndex] == 0 || pCur.aContextArray[nPrevIndex][nCurIndex] == 0)
            return 0.000001;

        int nPrevCurConFreq = pCur.aContextArray[nPrevIndex][nCurIndex];
        int nPrevFreq = pCur.aTagFreq[nPrevIndex];

        //0.9 and 0.1 is a value based experience
        return 0.9 * (double)nPrevCurConFreq / (double)nPrevFreq + 0.1 * (double)
                nPrevFreq / (double)pCur.nTotalFreq;
    }

    public void ReleaseContextStat()
    {
        m_pContext = null;
        m_pSymbolTable = null;
    }

    /**
     * 一个上下文信息
     */
    public static class ContextItem
    {
        /**
         * 目前看来恒为0，无用
         */
        public int nKey; //The key word
        /**
         * 转移矩阵
         */
        public int[][] aContextArray; //The context array
        /**
         * 每个标签出现的次数
         */
        public int[] aTagFreq; //The total number a tag appears
        /**
         * 标签出现的总次数
         */
        public int nTotalFreq; //The total number of all the tags

        /**
         * 下一个上下文
         */
        public ContextItem next; //The chain pointer to next Context
    }
}
