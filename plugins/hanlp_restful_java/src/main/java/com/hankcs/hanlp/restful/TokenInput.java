/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2020-12-27 12:09 AM</create-date>
 *
 * <copyright file="TokenInput.java">
 * Copyright (c) 2020, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful;

/**
 * @author hankcs
 */
public class TokenInput extends BaseInput
{
    public String[][] tokens;

    public TokenInput(String[][] tokens, String[] tasks, String[] skipTasks, String language)
    {
        super(tasks, skipTasks, language);
        this.tokens = tokens;
    }
}
