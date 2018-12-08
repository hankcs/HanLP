package com.hankcs.hanlp.collection.trie.datrie;

import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;

import java.io.*;
import java.util.ArrayList;

/**
 * 动态数组
 */
public class IntArrayList implements Serializable, ICacheAble
{
    private static final long serialVersionUID = 1908530358259070518L;
    private int[] data;
    /**
     * 实际size
     */
    private int size;
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

    public IntArrayList()
    {
        this(1024);
    }

    public IntArrayList(int capacity)
    {
        this(capacity, 10240);
    }

    public IntArrayList(int capacity, int linearExpandFactor)
    {
        this.data = new int[capacity];
        this.size = 0;
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
        if (this.size == this.data.length)
        {
            expand();
        }
        this.data[this.size] = element;
        this.size += 1;
    }

    /**
     * 去掉多余的buffer
     */
    public void loseWeight()
    {
        if (size == data.length)
        {
            return;
        }
        int[] newData = new int[size];
        System.arraycopy(this.data, 0, newData, 0, size);
        this.data = newData;
    }

    public int size()
    {
        return this.size;
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
        --size;
    }

    public int getLast()
    {
        return data[size - 1];
    }

    public void setLast(int value)
    {
        data[size - 1] = value;
    }

    public int pop()
    {
        return data[--size];
    }

    @Override
    public void save(DataOutputStream out) throws IOException
    {
        out.writeInt(size);
        for (int i = 0; i < size; i++)
        {
            out.writeInt(data[i]);
        }
        out.writeInt(linearExpandFactor);
        out.writeBoolean(exponentialExpanding);
        out.writeDouble(exponentialExpandFactor);
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        if (byteArray == null)
        {
            return false;
        }
        size = byteArray.nextInt();
        data = new int[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = byteArray.nextInt();
        }
        linearExpandFactor = byteArray.nextInt();
        exponentialExpanding = byteArray.nextBoolean();
        exponentialExpandFactor = byteArray.nextDouble();
        return true;
    }

    private void writeObject(ObjectOutputStream out) throws IOException
    {
        loseWeight();
        out.writeInt(size);
        out.writeObject(data);
        out.writeInt(linearExpandFactor);
        out.writeBoolean(exponentialExpanding);
        out.writeDouble(exponentialExpandFactor);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
    {
        size = in.readInt();
        data = (int[]) in.readObject();
        linearExpandFactor = in.readInt();
        exponentialExpanding = in.readBoolean();
        exponentialExpandFactor = in.readDouble();
    }

    @Override
    public String toString()
    {
        ArrayList<Integer> head = new ArrayList<Integer>(20);
        for (int i = 0; i < Math.min(size, 20); ++i)
        {
            head.add(data[i]);
        }
        return head.toString();
    }
}
