/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone.details;

/**
 * 整型动态数组<br>
 * Memory management of resizable array.
 *
 * @author
 */
class AutoIntPool
{
    int[] getBuffer()
    {
        return _buf;
    }

    int get(int id)
    {
        return _buf[id];
    }

    void set(int id, int value)
    {
        _buf[id] = value;
    }

    boolean empty()
    {
        return (_size == 0);
    }

    int size()
    {
        return _size;
    }

    void clear()
    {
        resize(0);
        _buf = null;
        _size = 0;
        _capacity = 0;
    }

    void add(int value)
    {
        if (_size == _capacity)
        {
            resizeBuf(_size + 1);
        }
        _buf[_size++] = value;
    }

    void deleteLast()
    {
        --_size;
    }

    void resize(int size)
    {
        if (size > _capacity)
        {
            resizeBuf(size);
        }
        _size = size;
    }

    void resize(int size, int value)
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

    void reserve(int size)
    {
        if (size > _capacity)
        {
            resizeBuf(size);
        }
    }

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
        int[] buf = new int[capacity];
        if (_size > 0)
        {
            System.arraycopy(_buf, 0, buf, 0, _size);
        }
        _buf = buf;
        _capacity = capacity;
    }

    private int[] _buf;
    private int _size;
    private int _capacity;
}
