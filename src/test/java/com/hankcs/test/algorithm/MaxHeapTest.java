package com.hankcs.test.algorithm;

import com.hankcs.hanlp.algorithm.MaxHeap;
import junit.framework.TestCase;

import java.util.Comparator;

public class MaxHeapTest extends TestCase
{
    final MaxHeap<Integer> heap = new MaxHeap<Integer>(5, new Comparator<Integer>()
    {
        @Override
        public int compare(Integer o1, Integer o2)
        {
            return o1.compareTo(o2);
        }
    });

    public void testAdd() throws Exception
    {
        heap.add(1);
        heap.add(3);
        heap.add(5);
        heap.add(7);
        heap.add(9);
        heap.add(8);
        heap.add(6);
        heap.add(4);
        heap.add(2);
        heap.add(0);
    }

    public void testAddAll() throws Exception
    {

    }

    public void testToList() throws Exception
    {
        testAdd();
        System.out.println(heap.toList());
    }
}