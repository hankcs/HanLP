/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone;

/**
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

    @Override
    public String toString()
    {
        return first + "=" + second;
    }
}
