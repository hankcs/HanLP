/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-29 4:49 PM</create-date>
 *
 * <copyright file="Pipe.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.pipe;

/**
 * 一截管道
 *
 * @param <I> 输入类型
 * @param <O> 输出类型
 * @author hankcs
 */
public interface Pipe<I, O>
{
    /**
     * 流经管道
     *
     * @param input 输入
     * @return 输出
     */
    O flow(I input);
}