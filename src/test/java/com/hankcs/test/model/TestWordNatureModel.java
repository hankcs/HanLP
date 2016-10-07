/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 15:33</create-date>
 *
 * <copyright file="TestModel.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.model;

import com.hankcs.hanlp.corpus.dependency.model.WordNatureWeightModelMaker;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.model.bigram.WordNatureDependencyModel;
import com.hankcs.hanlp.model.maxent.MaxEntModel;
import com.hankcs.hanlp.utility.Predefine;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestWordNatureModel extends TestCase
{
    String PATH = "data/model/dependency/test.txt";
    public void testLoad() throws Exception
    {
//        System.out.println(WordNatureDependencyModel.get("鼓励@" + WordNatureWeightModelMaker.wrapTag("v")));
//        System.out.println(WordNatureDependencyModel.get("鼓励@也是"));
//        System.out.println(WordNatureDependencyModel.get("鼓励@##核心##"));
//        System.out.println(WordNatureDependencyModel.get("方法论@123"));
//        System.out.println(WordNatureDependencyModel.get("方略@" + WordNatureWeightModelMaker.wrapTag("vshi")));
    }

    public void testMaxEntModel() throws Exception
    {
        MaxEntModel model = MaxEntModel.create(PATH);
        model = MaxEntModel.create(ByteArray.createByteArray(PATH + Predefine.BIN_EXT));
        String[] contexts = new String[]{"Rainy", "Sad"};
        System.out.println(model.predict(contexts));
    }
}
