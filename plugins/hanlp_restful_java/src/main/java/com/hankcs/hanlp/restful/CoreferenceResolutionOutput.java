/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2021-10-16 4:43 PM</create-date>
 *
 * <copyright file="CoreferenceResolutionOutput.java">
 * Copyright (c) 2021, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A data class for coreference resolution
 *
 * @author hankcs
 */
public class CoreferenceResolutionOutput
{
    public List<Set<Span>> clusters;
    public ArrayList<String> tokens;

    public CoreferenceResolutionOutput(List<Set<Span>> clusters, ArrayList<String> tokens)
    {
        this.clusters = clusters;
        this.tokens = tokens;
    }
}
