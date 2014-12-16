/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/14 19:24</create-date>
 *
 * <copyright file="DynamicArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.dynamicarray;

import java.util.ArrayList;
import java.util.List;

/**
 * @author He Han
 */
public abstract class DynamicArray<T>
{
    protected ChainItem<T> pHead;  //The head pointer of array chain
    public int ColumnCount, RowCount;  //The row and col of the array


    public DynamicArray()
    {
        pHead = null;
        RowCount = 0;
        ColumnCount = 0;
    }
    public int ItemCount()
    {
            ChainItem<T> pCur = pHead;
            int nCount = 0;
            while (pCur != null)
            {
                nCount++;
                pCur = pCur.next;
            }
            return nCount;
    }

    /**
     * 查找行、列值为nRow, nCol的结点
     * @param nRow
     * @param nCol
     * @return
     */
    public ChainItem<T> GetElement(int nRow, int nCol)
    {
        ChainItem<T> pCur = pHead;

        while (pCur != null && !(pCur.col == nCol && pCur.row == nRow))
            pCur = pCur.next;

        return pCur;
    }

    /**
     * 设置或插入一个新的结点
     * @param nRow
     * @param nCol
     * @param content
     */
    public abstract void SetElement(int nRow, int nCol, T content);

    /**
     * Return the head element of ArrayChain
     * @return
     */
    public ChainItem<T> GetHead()
    {
        return pHead;
    }

    /**
     * Get the tail Element buffer and return the count of elements
     * @param pTailRet
     * @return
     */
    public int GetTail(ChainItem<T> pTailRet)
    {
        ChainItem<T> pCur = pHead, pPrev = null;
        int nCount = 0;
        while (pCur != null)
        {
            nCount++;
            pPrev = pCur;
            pCur = pCur.next;
        }
        pTailRet.copy(pPrev);
        return nCount;
    }

    /**
     *Set Empty
     */
    public void SetEmpty()
    {
        pHead = null;
        ColumnCount = 0;
        RowCount = 0;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        ChainItem<T> pCur = pHead;

        while (pCur != null)
        {
            sb.append(String.format("row:%d,  col:%d,  ", pCur.row, pCur.col));
            sb.append(pCur.Content);
            sb.append("\r\n");
            pCur = pCur.next;
        }

        return sb.toString();
    }


    public List<ChainItem<T>> ToListItems()
    {
        List<ChainItem<T>> result = new ArrayList<ChainItem<T>>();

        ChainItem<T> pCur = pHead;
        while (pCur != null)
        {
            result.add(pCur);
            pCur = pCur.next;
        }

        return result;
    }
}
