/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/17 13:47</create-date>
 *
 * <copyright file="WordLinkedArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort.Path;

/**
 * 单词链表
 * @author hankcs
 */
public class WordLinkedArray
{
    public WordNode first = null;
    public WordNode last = null;
    public int Count = 0;

    public void AppendNode(WordNode node)
    {
        if (first == null && last == null)
        {
            first = node;
            last = node;
        } else
        {
            last.next = node;
            last = node;
        }

        Count++;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        WordNode cur = first;
        while (cur != null)
        {
            sb.append(cur.theWord.sWord);
            cur = cur.next;
        }

        return sb.toString();
    }

    /**
     * 单词节点
     */
    static public class WordNode
    {
        /**
         * 词图中的行
         */
        public int row;
        /**
         * 词图中的列
         */
        public int col;
        /**
         * 分词结果
         */
        public WordResult theWord;
        /**
         * 词图中的单词
         */
        public String sWordInSegGraph;

        /**
         * 下一个节点
         */
        public WordNode next;

        public WordNode(int row, int col, WordResult theWord, String sWordInSegGraph)
        {
            this.row = row;
            this.col = col;
            this.theWord = theWord;
            this.sWordInSegGraph = sWordInSegGraph;
        }
    }
}
