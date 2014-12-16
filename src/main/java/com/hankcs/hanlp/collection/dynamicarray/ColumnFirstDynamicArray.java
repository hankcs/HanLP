/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/14 19:46</create-date>
 *
 * <copyright file="ColumnFirstDynamicArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.dynamicarray;

/**
 * @author He Han
 */
public class ColumnFirstDynamicArray<T> extends DynamicArray<T>
{
    /**
     * 查找列为 nCol 的第一个结点
     *
     * @param nCol
     * @return
     */
    public ChainItem<T> GetFirstElementOfCol(int nCol)
    {
        ChainItem<T> pCur = pHead;

        while (pCur != null && pCur.col != nCol)
            pCur = pCur.next;

        return pCur;
    }

    /**
     * 从 startFrom 处向后查找列为 nCol 的第一个结点
     *
     * @param nCol
     * @param startFrom
     * @return
     */
    public ChainItem<T> GetFirstElementOfCol(int nCol, ChainItem<T> startFrom)
    {
        ChainItem<T> pCur = startFrom;

        while (pCur != null && pCur.col != nCol)
            pCur = pCur.next;

        return pCur;
    }


    /**
     * 设置或插入一个新的结点
     *
     * @param nRow
     * @param nCol
     * @param content
     */
    public void SetElement(int nRow, int nCol, T content)
    {
        ChainItem<T> pCur = pHead, pPre = null, pNew;  //The pointer of array chain

        if (nRow > RowCount)//Set the array row
            RowCount = nRow;

        if (nCol > ColumnCount)//Set the array col
            ColumnCount = nCol;

        while (pCur != null && (pCur.col < nCol || (pCur.col == nCol && pCur.row < nRow)))
        {
            pPre = pCur;
            pCur = pCur.next;
        }

        if (pCur != null && pCur.row == nRow && pCur.col == nCol)//Find the same position
            pCur.Content = content;//Set the value
        else
        {
            pNew = new ChainItem<T>();//malloc a new node
            pNew.col = nCol;
            pNew.row = nRow;
            pNew.Content = content;

            pNew.next = pCur;

            if (pPre == null)//link pNew after the pPre
                pHead = pNew;
            else
                pPre.next = pNew;
        }
    }

}
