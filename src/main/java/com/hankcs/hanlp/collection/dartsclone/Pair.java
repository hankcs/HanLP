/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone;

/**
 * 模拟C++中的pair，也兼容JavaFX中的Pair
 * @author manabe
 */
public class Pair<T, U>
{
    public final T first;
    public final U second;

    public Pair(T first, U second)
    {
        this.first = first;
        this.second = second;
    }

    public T getFirst()
    {
        return first;
    }

    public T getKey()
    {
        return first;
    }

    public U getSecond()
    {
        return second;
    }

    public U getValue()
    {
        return second;
    }

    @Override
    public String toString()
    {
        return first + "=" + second;
    }
}
