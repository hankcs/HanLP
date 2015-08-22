/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/7 19:13</create-date>
 *
 * <copyright file="DemoNLPSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.TraditionalChineseTokenizer;

import java.util.List;

/**
 * 繁体中文分词
 *
 * @author hankcs
 */
public class DemoTraditionalChineseSegment
{
    public static void main(String[] args)
    {
        List<Term> termList = TraditionalChineseTokenizer.segment("大衛貝克漢不僅僅是名著名球員，球場以外，其妻為前" +
                                                                          "辣妹合唱團成員維多利亞·碧咸，亦由於他擁有" +
                                                                          "突出外表、百變髮型及正面的形象，以至自己" +
                                                                          "品牌的男士香水等商品，及長期擔任運動品牌" +
                                                                          "Adidas的代言人，因此對大眾傳播媒介和時尚界" +
                                                                          "等方面都具很大的影響力，在足球圈外所獲得的" +
                                                                          "認受程度可謂前所未見。");
        System.out.println(termList);

        termList = TraditionalChineseTokenizer.segment("（中央社記者黃巧雯台北20日電）外需不振，影響接單動能，經濟部今天公布7月外銷訂單金額362.9億美元，年減5%，" +
                                                               "連續4個月衰退，減幅較6月縮小。1040820\n");
        System.out.println(termList);

        termList = TraditionalChineseTokenizer.segment("中央社记者黄巧雯台北20日电");
        System.out.println(termList);
    }
}
