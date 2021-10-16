/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2021-10-16 4:26 PM</create-date>
 *
 * <copyright file="Span.java">
 * Copyright (c) 2021, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful;

import java.util.Objects;

/**
 * A common data format to represent a span.
 *
 * @author hankcs
 */
public class Span
{
    /**
     * The raw form of a span, which can be either a token, an entitiy or a mention etc.
     */
    public String form;
    /**
     * The inclusive beginning offset of a span.
     */
    public int begin;
    /**
     * The exclusive ending offset of a span.
     */
    public int end;

    public Span(String form, int begin, int end)
    {
        this.form = form;
        this.begin = begin;
        this.end = end;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Span span = (Span) o;
        return begin == span.begin &&
                end == span.end &&
                form.equals(span.form);
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(form, begin, end);
    }

    @Override
    public String toString()
    {
        return String.format("[%d, %d) = %s", begin, end, form);
    }
}
