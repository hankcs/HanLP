/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2020-12-27 12:09 AM</create-date>
 *
 * <copyright file="SentenceInput.java">
 * Copyright (c) 2020, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful;

/**
 * @author hankcs
 */
public class SentenceInput extends BaseInput
{
    public String[] text;

    public SentenceInput(String[] text, String[] tasks, String[] skipTasks, String language)
    {
        super(tasks, skipTasks, language);
        this.text = text;
    }
}
