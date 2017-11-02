/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/10 PM5:35</create-date>
 *
 * <copyright file="ITokenizer.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.tokenizers;

import java.io.Serializable;

/**
 * @author hankcs
 */
public interface ITokenizer extends Serializable
{
    String[] segment(String text);
}
