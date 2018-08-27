/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-18 11:11 PM</create-date>
 *
 * <copyright file="DemoTextClustering.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.mining.cluster.ClusterAnalyzer;

/**
 * @author hankcs
 */
public class DemoTextClustering
{
    public static void main(String[] args)
    {
        ClusterAnalyzer<String> analyzer = new ClusterAnalyzer<String>();
        analyzer.addDocument("赵一", "流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 蓝调, 蓝调, 蓝调, 蓝调, 蓝调, 蓝调, 摇滚, 摇滚, 摇滚, 摇滚");
        analyzer.addDocument("钱二", "爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲");
        analyzer.addDocument("张三", "古典, 古典, 古典, 古典, 民谣, 民谣, 民谣, 民谣");
        analyzer.addDocument("李四", "爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 金属, 金属, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲");
        analyzer.addDocument("王五", "流行, 流行, 流行, 流行, 摇滚, 摇滚, 摇滚, 嘻哈, 嘻哈, 嘻哈");
        analyzer.addDocument("马六", "古典, 古典, 古典, 古典, 古典, 古典, 古典, 古典, 摇滚");
        System.out.println(analyzer.kmeans(3));
        System.out.println(analyzer.repeatedBisection(3));
        System.out.println(analyzer.repeatedBisection(1.0)); // 自动判断聚类数量k
    }
}
