/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/25 17:55</create-date>
 * <author>Eric Hettiaratchi</author>
 * <email>erichettiaratchi@gmail.com</email>
 * <create-date>2019/07/24 18:31</create-date>
 *
 * <copyright file="ByteUtil.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2019, 上海林原信息科技有限公司. All Right Reserved, http://www * .linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import org.junit.Assert;
import org.junit.Test;

public class ByteUtilTest {

    @Test
    public void testBytesToChar() {
        Assert.assertEquals('Ă', ByteUtil.bytesToChar(new byte[]{1, 2}));
    }

    @Test
    public void testBytesToDouble() {
        Assert.assertEquals(8.20788039913184E-304,
            ByteUtil.bytesToDouble(new byte[]{1, 2, 3, 4, 5, 6, 7, 8}), 0);
    }

    @Test
    public void testBytesHighFirstToDouble() {
        Assert.assertEquals(8.20788039913184E-304,
            ByteUtil.bytesHighFirstToDouble(
                new byte[]{1, 2, 3, 4, 5, 6, 7, 8}, 0), 0);
        Assert.assertEquals(5.678932010640861E-299,
            ByteUtil.bytesHighFirstToDouble(
                new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1), 0);
    }

    @Test
    public void testBytesToFloat() {
        Assert.assertEquals(2.387939260590663E-38,
            ByteUtil.bytesToFloat(new byte[]{1, 2, 3, 4}), 0);
    }

    @Test
    public void testBytesToLong() {
        Assert.assertEquals(72623859790382856L,
            ByteUtil.bytesToLong(new byte[]{1, 2, 3, 4, 5, 6, 7, 8}));
    }

    @Test
    public void testBytesHighFirstToLong() {
        Assert.assertEquals(72623859790382856L,
            ByteUtil.bytesHighFirstToLong(new byte[]{1, 2, 3, 4, 5, 6, 7, 8}));
    }

    @Test
    public void testCharToBytes() {
        Assert.assertArrayEquals(new byte[]{0, 97, 0, 0, 0, 0, 0, 0},
            ByteUtil.charToBytes('a'));
    }

    @Test
    public void testDoubleToBytes() {
        Assert.assertArrayEquals(new byte[]{64, 4, 0, 0, 0, 0, 0, 0},
            ByteUtil.doubleToBytes(2.5));
    }

    @Test
    public void testFloatToBytes() {
        Assert.assertArrayEquals(new byte[]{64, 89, -103, -102},
            ByteUtil.floatToBytes(3.4f));
    }

    @Test
    public void testIntToBytes() {
        Assert.assertArrayEquals(new byte[]{0, 0, 0, 4},
            ByteUtil.intToBytes(4));
    }

    @Test
    public void testLongToBytes() {
        Assert.assertArrayEquals(new byte[]{0, 0, 0, 0, 0, 0, 0, 2},
            ByteUtil.longToBytes(2L));
    }

    @Test
    public void testBytesToInt() {
        Assert.assertEquals(67305985,
            ByteUtil.bytesToInt(new byte[]{1, 2, 3, 4, 5}, 0));
        Assert.assertEquals(84148994,
            ByteUtil.bytesToInt(new byte[]{1, 2, 3, 4, 5}, 1));
    }

    @Test
    public void testBytesHighFirstToInt() {
        Assert.assertEquals(16909060,
            ByteUtil.bytesHighFirstToInt(new byte[]{1, 2, 3, 4, 5}, 0));
        Assert.assertEquals(33752069,
            ByteUtil.bytesHighFirstToInt(new byte[]{1, 2, 3, 4, 5}, 1));
    }

    @Test
    public void testBytesHighFirstToChar() {
        Assert.assertEquals('Ă',
            ByteUtil.bytesHighFirstToChar(new byte[]{1, 2, 3, 4, 5}, 0));
        Assert.assertEquals('ȃ',
            ByteUtil.bytesHighFirstToChar(new byte[]{1, 2, 3, 4, 5}, 1));
    }

    @Test
    public void testBytesHighFirstToFloat() {
        Assert.assertEquals(2.387939260590663E-38,
            ByteUtil.bytesHighFirstToFloat(new byte[]{1, 2, 3, 4, 5}, 0), 0);
        Assert.assertEquals(9.625513546253311E-38,
            ByteUtil.bytesHighFirstToFloat(new byte[]{1, 2, 3, 4, 5}, 1), 0);
    }

    @Test
    public void testConvertTwoCharToInt() {
        Assert.assertEquals(6619234,
            ByteUtil.convertTwoCharToInt('e', 'b'));
        Assert.assertEquals(4522082,
            ByteUtil.convertTwoCharToInt('E', 'b'));
    }

    @Test
    public void testConvertIntToTwoChar() {
        Assert.assertArrayEquals(new char[]{'\u0000', 68},
            ByteUtil.convertIntToTwoChar(68));
        Assert.assertArrayEquals(new char[]{'\u0000', 78},
            ByteUtil.convertIntToTwoChar(78));
    }
}
