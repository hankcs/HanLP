/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone.details;

/**
 * Bit向量，类似于C++中的bitset
 * @author
 */
class BitVector
{
    /**
     * 获取某一位的比特
     * @param id 位
     * @return 比特是1还是0
     */
    boolean get(int id)
    {
        return (_units.get(id / UNIT_SIZE) >>> (id % UNIT_SIZE) & 1) == 1;
    }

    /**
     * 设置某一位的比特
     * @param id 位
     * @param bit 比特
     */
    void set(int id, boolean bit)
    {
        if (bit)
        {
            _units.set(id / UNIT_SIZE, _units.get(id / UNIT_SIZE)
                    | 1 << (id % UNIT_SIZE));
        }
    }

    /**
     *
     * @param id
     * @return
     */
    int rank(int id)
    {
        int unit_id = id / UNIT_SIZE;
        return _ranks[unit_id] + popCount(_units.get(unit_id)
                                                  & (~0 >>> (UNIT_SIZE - (id % UNIT_SIZE) - 1)));
    }

    /**
     * 是否为空
     * @return
     */
    boolean empty()
    {
        return _units.empty();
    }

    /**
     * 1的数量
     * @return
     */
    int numOnes()
    {
        return _numOnes;
    }

    /**
     * 大小
     * @return
     */
    int size()
    {
        return _size;
    }

    /**
     * 在末尾追加
     */
    void append()
    {
        if ((_size % UNIT_SIZE) == 0)
        {
            _units.add(0);
        }
        ++_size;
    }

    /**
     * 构建
     */
    void build()
    {
        _ranks = new int[_units.size()];

        _numOnes = 0;
        for (int i = 0; i < _units.size(); ++i)
        {
            _ranks[i] = _numOnes;
            _numOnes += popCount(_units.get(i));
        }
    }

    /**
     * 清空
     */
    void clear()
    {
        _units.clear();
        _ranks = null;
    }

    /**
     * 整型大小
     */
    private static final int UNIT_SIZE = 32; // sizeof(int) * 8

    /**
     * 1的数量
     * @param unit
     * @return
     */
    private static int popCount(int unit)
    {
        unit = ((unit & 0xAAAAAAAA) >>> 1) + (unit & 0x55555555);
        unit = ((unit & 0xCCCCCCCC) >>> 2) + (unit & 0x33333333);
        unit = ((unit >>> 4) + unit) & 0x0F0F0F0F;
        unit += unit >>> 8;
        unit += unit >>> 16;
        return unit & 0xFF;
    }

    /**
     * 储存空间
     */
    private AutoIntPool _units = new AutoIntPool();
    /**
     * 是每个元素的1的个数的累加
     */
    private int[] _ranks;
    /**
     * 1的数量
     */
    private int _numOnes;
    /**
     * 大小
     */
    private int _size;
}
