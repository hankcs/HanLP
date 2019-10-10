/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-26 2:27 PM</create-date>
 *
 * <copyright file="ExportCorpusToCRF.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch06;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.crf.CRFSegmenter;

import java.io.IOException;

/**
 * 《自然语言处理入门》6.3 条件随机场工具包
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class CrfppTrainHanLPLoad
{
    public static final String TXT_CORPUS_PATH = "data/test/my_cws_corpus.txt";
    public static final String TSV_CORPUS_PATH = TXT_CORPUS_PATH + ".tsv";
    public static final String TEMPLATE_PATH = "data/test/cws-template.txt";
    public static final String CRF_MODEL_PATH = "data/test/crf-cws-model";
    public static final String CRF_MODEL_TXT_PATH = "data/test/crf-cws-model.txt";

    public static void main(String[] args) throws IOException
    {
        if (IOUtil.isFileExisted(CRF_MODEL_TXT_PATH))
        {
            CRFSegmenter segmenter = new CRFSegmenter(CRF_MODEL_TXT_PATH);
            System.out.println(segmenter.segment("商品和服务"));
        }
        else
        {
            CRFSegmenter segmenter = new CRFSegmenter(null); // 创建空白分词器
            segmenter.convertCorpus(TXT_CORPUS_PATH, TSV_CORPUS_PATH); // 执行转换
            segmenter.dumpTemplate(TEMPLATE_PATH);
            System.out.printf("语料已转换为 %s ，特征模板已导出为 %s\n", TSV_CORPUS_PATH, TEMPLATE_PATH);
            System.out.printf("请安装CRF++后执行 crf_learn -f 3 -c 4.0 %s %s %s -t\n", TEMPLATE_PATH, TSV_CORPUS_PATH, CRF_MODEL_PATH);
            System.out.printf("或者执行移植版 java -cp hanlp.jar com.hankcs.hanlp.model.crf.crfpp.crf_learn -f 3 -c 4.0 %s %s %s -t\n", TEMPLATE_PATH, TSV_CORPUS_PATH, CRF_MODEL_PATH);
        }
    }
}
