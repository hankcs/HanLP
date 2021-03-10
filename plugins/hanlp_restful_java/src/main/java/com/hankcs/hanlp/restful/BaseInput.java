/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2020-12-27 12:07 AM</create-date>
 *
 * <copyright file="Input.java">
 * Copyright (c) 2020, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful;

/**
 * @author hankcs
 */
public class BaseInput
{
    public String[] tasks;
    public String[] skip_tasks;
    public String language;

    public BaseInput(String[] tasks, String[] skipTasks, String language)
    {
        this.tasks = tasks;
        this.skip_tasks = skipTasks;
        this.language = language;
    }
}
