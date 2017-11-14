/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/10 PM4:21</create-date>
 *
 * <copyright file="IClassificationCorpus.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.corpus;

import com.hankcs.hanlp.classification.tokenizers.ITokenizer;

import java.io.IOException;
import java.util.Map;

/**
 * 文本分类数据集接口
 *
 * @author hankcs
 */
public interface IDataSet extends Iterable<Document>
{
    /**
     * 加载数据集
     *
     * @param folderPath 分类语料的根目录.目录必须满足如下结构:<br>
     *                   根目录<br>
     *                   ├── 分类A<br>
     *                   │   └── 1.txt<br>
     *                   │   └── 2.txt<br>
     *                   │   └── 3.txt<br>
     *                   ├── 分类B<br>
     *                   │   └── 1.txt<br>
     *                   │   └── ...<br>
     *                   └── ...<br>
     *                   文件不一定需要用数字命名,也不需要以txt作为后缀名,但一定需要是文本文件.
     * @return
     * @throws IllegalArgumentException
     * @throws IOException
     */
    IDataSet load(String folderPath) throws IllegalArgumentException, IOException;
    IDataSet load(String folderPath, double rate) throws IllegalArgumentException, IOException;

    /**
     * 加载数据集
     *
     * @param folderPath  分类语料的根目录.目录必须满足如下结构:<br>
     *                    根目录<br>
     *                    ├── 分类A<br>
     *                    │   └── 1.txt<br>
     *                    │   └── 2.txt<br>
     *                    │   └── 3.txt<br>
     *                    ├── 分类B<br>
     *                    │   └── 1.txt<br>
     *                    │   └── ...<br>
     *                    └── ...<br>
     *                    文件不一定需要用数字命名,也不需要以txt作为后缀名,但一定需要是文本文件.
     * @param charsetName 文件编码
     * @return
     * @throws IllegalArgumentException
     * @throws IOException
     */
    IDataSet load(String folderPath, String charsetName) throws IllegalArgumentException, IOException;
    IDataSet load(String folderPath, String charsetName, double percentage) throws IllegalArgumentException, IOException;

    /**
     * 往训练集中加入一个文档
     *
     * @param category 分类
     * @param text     文本
     * @return
     */
    Document add(String category, String text);

    /**
     * 利用本数据集的词表和类目表将文本形式的文档转换为内部通用的文档
     *
     * @param category
     * @param text
     * @return
     */
    Document convert(String category, String text);

    /**
     * 设置分词器
     *
     * @param tokenizer
     * @return
     */
    IDataSet setTokenizer(ITokenizer tokenizer);

    /**
     * 数据集的样本大小
     *
     * @return
     */
    int size();

    /**
     * 获取分词器
     *
     * @return
     */
    ITokenizer getTokenizer();

    /**
     * 获取类目表
     *
     * @return
     */
    Catalog getCatalog();

    /**
     * 获取词表
     *
     * @return
     */
    Lexicon getLexicon();

    /**
     * 清空数据集
     */
    void clear();

    /**
     * 是否是测试集
     *
     * @return
     */
    boolean isTestingDataSet();

    IDataSet add(Map<String, String[]> testingDataSet);

    IDataSet shrink(int[] idMap);

//    /**
//     * 分割数据集
//     * @param rate 得到新数据集占原数据集的大小比率,比如原本有10个文档,rate=0.1,则新数据集有1个文档,旧数据集变成了9个文档
//     * @return 新数据集
//     * @throws IllegalArgumentException
//     */
//    IDataSet spilt(double rate) throws IllegalArgumentException;
}