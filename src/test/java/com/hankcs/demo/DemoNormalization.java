/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/18 16:08</create-date>
 *
 * <copyright file="DemoNormalization.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CustomDictionary;

/**
 * 演示正规化字符配置项的效果（繁体->简体，全角->半角，大写->小写）。
 * 该配置项位于hanlp.properties中，通过Normalization=true来开启
 * 切换配置后必须删除CustomDictionary.txt.bin缓存，否则只影响动态插入的新词。
 *
 * @author hankcs
 */
public class DemoNormalization
{
    public static void main(String[] args)
    {
        HanLP.Config.Normalization = true;
        CustomDictionary.insert("爱听4G", "nz 1000");
        System.out.println(HanLP.segment("爱听4g"));
        System.out.println(HanLP.segment("爱听4G"));
        System.out.println(HanLP.segment("爱听４G"));
        System.out.println(HanLP.segment("爱听４Ｇ"));
        System.out.println(HanLP.segment("愛聽４Ｇ"));
    }
}
