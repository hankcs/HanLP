/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 下午7:33</create-date>
 *
 * <copyright file="NERecognizer.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.lexical;

import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;

/**
 * 命名实体识别接口
 *
 * @author hankcs
 */
public interface NERecognizer
{
    /**
     * 命名实体识别
     *
     * @param wordArray 单词
     * @param posArray  词性
     * @return BMES-NER标签
     */
    String[] recognize(String[] wordArray, String[] posArray);

    NERTagSet getNERTagSet();
}
