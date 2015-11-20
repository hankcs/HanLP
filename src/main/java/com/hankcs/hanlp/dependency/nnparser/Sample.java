/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/1 21:27</create-date>
 *
 * <copyright file="Sample.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import java.util.List;

/**
 * @author hankcs
 */
public class Sample
{
    List<Integer> attributes;  //! sparse vector of attributes
    List<Double> classes;  //! dense vector of classes

    public Sample()
    {
    }

    public Sample(List<Integer> attributes, List<Double> classes)
    {
        this.attributes = attributes;
        this.classes = classes;
    }
}
