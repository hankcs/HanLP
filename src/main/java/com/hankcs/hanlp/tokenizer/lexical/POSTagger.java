/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 下午7:36</create-date>
 *
 * <copyright file="POSTagger.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.lexical;

import java.util.List;

/**
 * @author hankcs
 */
public interface POSTagger
{
    String[] tag(String... words);
    String[] tag(List<String> wordList);
}
