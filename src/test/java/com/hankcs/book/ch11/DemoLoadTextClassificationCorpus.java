/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-01-03 6:56 PM</create-date>
 *
 * <copyright file="DemoLoadTextClassificationCorpus.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch11;

import com.hankcs.hanlp.classification.corpus.AbstractDataSet;
import com.hankcs.hanlp.classification.corpus.Document;
import com.hankcs.hanlp.classification.corpus.FileDataSet;
import com.hankcs.hanlp.classification.corpus.MemoryDataSet;

import java.io.IOException;
import java.util.List;

import static com.hankcs.demo.DemoTextClassification.CORPUS_FOLDER;


/**
 * 《自然语言处理入门》11.2 文本分类语料库
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 * 演示加载文本分类语料库
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoLoadTextClassificationCorpus
{
    public static void main(String[] args) throws IOException
    {
        AbstractDataSet dataSet = new MemoryDataSet(); // ①将数据集加载到内存中
        dataSet.load(CORPUS_FOLDER); // ②加载data/test/搜狗文本分类语料库迷你版
        dataSet.add("自然语言处理", "自然语言处理很有趣"); // ③新增样本
        List<String> allClasses = dataSet.getCatalog().getCategories(); // ④获取标注集
        System.out.printf("标注集：%s\n", allClasses);
        for (Document document : dataSet)
        {
            System.out.println("第一篇文档的类别：" + allClasses.get(document.category));
            break;
        }
    }
}
