/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2022-04-13 8:57 AM</create-date>
 *
 * <copyright file="MeaningRepresentation.java">
 * Copyright (c) 2022, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful.mrp;

/**
 * Graph-based meaning representation.
 *
 * @author hankcs
 */
public class MeaningRepresentation
{
    public String id;
    public String input;
    public Node[] nodes;
    public Edge[] edges;
    public String[] tops;
    public String framework;
}
