/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 13:49</create-date>
 *
 * <copyright file="DemoWordDistance.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.dictionary.CoreSynonymDictionary;
import com.hankcs.hanlp.dictionary.common.CommonSynonymDictionary;

/**
 * 语义距离
 * @author hankcs
 */
public class DemoWordDistance
{
    public static void main(String[] args)
    {
        String apple = "苹果";
        String banana = "香蕉";
        String bike = "自行车";
        CommonSynonymDictionary.SynonymItem synonymApple = CoreSynonymDictionary.get(apple);
        CommonSynonymDictionary.SynonymItem synonymBanana = CoreSynonymDictionary.get(banana);
        CommonSynonymDictionary.SynonymItem synonymBike = CoreSynonymDictionary.get(bike);
        System.out.println(apple + " " + banana + " 之间的距离是 " + synonymApple.distance(synonymBanana));
        System.out.println(apple + " " + bike + " 之间的距离是 " + synonymApple.distance(synonymBike));
    }
}
