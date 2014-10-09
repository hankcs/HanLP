/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 23:56</create-date>
 *
 * <copyright file="NR.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;


import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;

/**
 * 人名识别的处理程序
 * @author hankcs
 */
public class NRTester
{
    public static void main(String[] args)
    {
        CorpusLoader.walk("D:\\Doc\\语料库\\2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                System.out.println(document);
            }
        });
    }

    public static void handle(Document document)
    {

    }
}
