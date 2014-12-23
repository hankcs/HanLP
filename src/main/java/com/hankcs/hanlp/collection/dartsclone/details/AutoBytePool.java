/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone.details;

/**
 * 动态数组<br>
 * Memory management of resizable array.
 *
 * @author
 */
class AutoBytePool
{
    /**
     * 获取缓冲区
     * @return 缓冲区
     */
    byte[] getBuffer()
    {
        return _buf;
    }

    /**
     * 取字节
     * @param id 字节下标
     * @return 字节
     */
    byte get(int id)
    {
        return _buf[id];
    }

    /**
     * 设置值
     * @param id 下标
     * @param value 值
     */
    void set(int id, byte value)
    {
        _buf[id] = value;
    }

    /**
     * 是否为空
     * @return true表示为空
     */
    boolean empty()
    {
        return (_size == 0);
    }

    /**
     * 缓冲区大小
     * @return 大小
     */
    int size()
    {
        return _size;
    }

    /**
     * 清空缓存
     */
    void clear()
    {
        resize(0);
        _buf = null;
        _size = 0;
        _capacity = 0;
    }

    /**
     * 在末尾加一个值
     * @param value 值
     */
    void add(byte value)
    {
        if (_size == _capacity)
        {
            resizeBuf(_size + 1);
        }
        _buf[_size++] = value;
    }

    /**
     * 将最后一个值去掉
     */
    void deleteLast()
    {
        --_size;
    }

    /**
     * 重设大小
     * @param size 大小
     */
    void resize(int size)
    {
        if (size > _capacity)
        {
            resizeBuf(size);
        }
        _size = size;
    }

    /**
     * 重设大小，并且在末尾加一个值
     * @param size 大小
     * @param value 值
     */
    void resize(int size, byte value)
    {
        if (size > _capacity)
        {
            resizeBuf(size);
        }
        while (_size < size)
        {
            _buf[_size++] = value;
        }
    }

    /**
     * 增加容量
     * @param size 容量
     */
    void reserve(int size)
    {
        if (size > _capacity)
        {
            resizeBuf(size);
        }
    }

    /**
     * 设置缓冲区大小
     * @param size 大小
     */
    private void resizeBuf(int size)
    {
        int capacity;
        if (size >= _capacity * 2)
        {
            capacity = size;
        }
        else
        {
            capacity = 1;
            while (capacity < size)
            {
                capacity <<= 1;
            }
        }
        byte[] buf = new byte[capacity];
        if (_size > 0)
        {
            System.arraycopy(_buf, 0, buf, 0, _size);
        }
        _buf = buf;
        _capacity = capacity;
    }

    /**
     * 缓冲区
     */
    private byte[] _buf;
    /**
     * 大小
     */
    private int _size;
    /**
     * 容量
     */
    private int _capacity;
}
