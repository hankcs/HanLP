/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/1 20:12</create-date>
 *
 * <copyright file="AdaOption.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.option;

/**
 * @author hankcs
 */
public class AdaOption
{
    double ada_eps;             //! Eps used in AdaGrad
    double ada_alpha;           //! Alpha used in AdaGrad
    double lambda;              //! TODO not known.
    double dropout_probability; //! The probability for dropout.
}
