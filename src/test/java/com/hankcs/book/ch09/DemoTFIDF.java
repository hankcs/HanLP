/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-09-17 12:22 AM</create-date>
 *
 * <copyright file="DemoTFIDF.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch09;

import com.hankcs.hanlp.mining.word.TfIdfCounter;

/**
 * 《自然语言处理入门》9.2 关键词提取
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoTFIDF
{
    public static void main(String[] args)
    {
        TfIdfCounter counter = new TfIdfCounter();
        counter.add("《女排夺冠》", "女排北京奥运会夺冠"); // 输入多篇文档
        counter.add("《羽毛球男单》", "北京奥运会的羽毛球男单决赛");
        counter.add("《女排》", "中国队女排夺北京奥运会金牌重返巅峰，观众欢呼女排女排女排！");
        
//        // 加载idf文件
//        counter.loadIdfFile("data/idf.txt");
        
        counter.compute(); // 输入完毕
        for (Object id : counter.documents()) // 根据每篇文档的TF-IDF提取关键词
        {
            System.out.println(id + " : " + counter.getKeywordsOf(id, 3));
        }
    }
}
