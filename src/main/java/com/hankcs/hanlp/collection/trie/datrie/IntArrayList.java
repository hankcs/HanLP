package com.hankcs.hanlp.collection.trie.datrie;

import java.io.Serializable;

/**
 * 动态数组
 */
public class IntArrayList implements Serializable
{
    private static final long serialVersionUID = 1908530358259070518L;
    private int[] data;
    private int count;
    /**
     * 线性递增
     */
    private int linearExpandFactor;

    public void setLinearExpandFactor(int linearExpandFactor)
    {
        this.linearExpandFactor = linearExpandFactor;
    }

    /**
     * 是否指数递增
     */
    private boolean exponentialExpanding = false;

    public boolean isExponentialExpanding()
    {
        return exponentialExpanding;
    }

    public void setExponentialExpanding(boolean multiplyExpanding)
    {
        this.exponentialExpanding = multiplyExpanding;
    }

    private double exponentialExpandFactor = 1.5;

    public double getExponentialExpandFactor()
    {
        return exponentialExpandFactor;
    }

    public void setExponentialExpandFactor(double exponentialExpandFactor)
    {
        this.exponentialExpandFactor = exponentialExpandFactor;
    }

    public IntArrayList(int size)
    {
        this(size, 10240);
    }

    public IntArrayList(int size, int linearExpandFactor)
    {
        this.data = new int[size];
        this.count = 0;
        this.linearExpandFactor = linearExpandFactor;
    }

    private void expand()
    {
        if (!exponentialExpanding)
        {
            int[] newData = new int[this.data.length + this.linearExpandFactor];
            System.arraycopy(this.data, 0, newData, 0, this.data.length);
            this.data = newData;
        }
        else
        {
            int[] newData = new int[(int) (this.data.length * exponentialExpandFactor)];
            System.arraycopy(this.data, 0, newData, 0, this.data.length);
            this.data = newData;
        }
    }

    /**
     * 在数组尾部新增一个元素
     *
     * @param element
     */
    public void append(int element)
    {
        if (this.count == this.data.length)
        {
            expand();
        }
        this.data[this.count] = element;
        this.count += 1;
    }

    /**
     * 去掉多余的buffer
     */
    public void loseWeight()
    {
        if (count == data.length)
        {
            return;
        }
        int[] newData = new int[count];
        System.arraycopy(this.data, 0, newData, 0, count);
        this.data = newData;
    }

    public int size()
    {
        return this.count;
    }

    public int getLinearExpandFactor()
    {
        return this.linearExpandFactor;
    }

    public void set(int index, int value)
    {
        this.data[index] = value;
    }

    public int get(int index)
    {
        return this.data[index];
    }

    public void removeLast()
    {
        --count;
    }

    public int getLast()
    {
        return data[count - 1];
    }

    public int pop()
    {
        return data[--count];
    }
}
