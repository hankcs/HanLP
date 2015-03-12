/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/3/12 20:18</create-date>
 *
 * <copyright file="DemoAddressExtraction.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.AddressTerm;

import java.util.LinkedList;

/**
 * 测试地址提取
 * @author hankcs
 */
public class DemoAddressExtraction
{
    public static void main(String[] args)
    {
        String text = "上海中山医院地址: " +
                "本部:上海市徐汇区枫林路180号;" +
                "东院:上海市徐汇区斜土路的1609号," +
                "延安西路分院:长宁区延安西路1474号;" +
                "电话:021-64041990(本部总机) " +
                "怎么走：本部行车路线(外地患者):火车站路线:从上海火车站出发,乘地铁4号线在东安路下车,步行至医院; " +
                "本部行车路线(本地患者):乘坐公交104路在小木桥路斜土路,927路在乌鲁木齐南路建国西路下车," +
                "43路,45路,49路,205路,218路,733路,806路,820路,864路,931路,957路,984路,985路,隧道二线,徐川专线,步行至医院";
        LinkedList<AddressTerm> addressTermLinkedList = HanLP.extractAddress(text);
        for (AddressTerm term : addressTermLinkedList)
        {
            System.out.printf("[%d:%d] = %s = %s\n", term.offset, term.offset + term.length(), term.word, term.detail);
        }
    }
}
