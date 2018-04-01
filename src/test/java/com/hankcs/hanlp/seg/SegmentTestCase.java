/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-20 下午12:19</create-date>
 *
 * <copyright file="SegmentTestCase.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg;

import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.Assert;
import junit.framework.TestCase;

import java.util.List;

/**
 * @author hankcs
 */
public class SegmentTestCase extends TestCase
{
    @SuppressWarnings("deprecation")
    public static void assertNoNature(List<Term> termList, Nature nature)
    {
        for (Term term : termList)
        {
            Assert.assertNotSame(nature, term.nature);
        }
    }

    @SuppressWarnings("deprecation")
    public static void assertSegmentationHas(List<Term> termList, String part)
    {
        StringBuilder sbSentence = new StringBuilder();
        for (Term term : termList)
        {
            sbSentence.append(term.word);
        }
        assertFalse(sbSentence.toString().contains(part));
    }

}
