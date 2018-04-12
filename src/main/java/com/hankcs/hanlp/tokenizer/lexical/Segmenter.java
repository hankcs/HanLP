/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 下午7:30</create-date>
 *
 * <copyright file="Segmenter.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.lexical;

import java.util.List;

/**
 * 分词器接口
 *
 * @author hankcs
 */
public interface Segmenter
{
    /**
     * 中文分词
     *
     * @param text 文本
     * @return 词语
     */
    List<String> segment(String text);
    void segment(String text, String normalized, List<String> output);
}
