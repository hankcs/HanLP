/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/14 21:36</create-date>
 *
 * <copyright file="Predefine.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * 一些预定义的静态全局变量
 */
public class Predefine
{
    public static final String CHINESE_NUMBERS = "零○〇一二两三四五六七八九十廿百千万亿壹贰叁肆伍陆柒捌玖拾佰仟";
    /**
     * hanlp.properties的路径，一般情况下位于classpath目录中。
     * 但在某些极端情况下（不标准的Java虚拟机，用户缺乏相关知识等），允许将其设为绝对路径
     */
    public static String HANLP_PROPERTIES_PATH;
    public final static double MIN_PROBABILITY = 1e-10;
    /**
     * 浮点数正则
     */
    public static final Pattern PATTERN_FLOAT_NUMBER = Pattern.compile("^(-?\\d+)(\\.\\d+)?$");

    public static String POSTFIX_SINGLE =
        "坝邦堡城池村单岛道堤店洞渡队峰府冈港阁宫沟国海号河湖环集江礁角街井郡坑口矿里岭楼路门盟庙弄牌派坡铺旗桥区渠泉山省市水寺塔台滩坛堂厅亭屯湾屋溪峡县线乡巷洋窑营屿园苑院闸寨站镇州庄族陂庵町";

    public final static String[] POSTFIX_MUTIPLE = {"半岛","草原","城区","大堤","大公国","大桥","地区",
        "帝国","渡槽","港口","高速公路","高原","公路","公园","共和国","谷地","广场",
        "国道","海峡","胡同","机场","集镇","教区","街道","口岸","码头","煤矿",
        "牧场","农场","盆地","平原","丘陵","群岛","沙漠","沙洲","山脉","山丘",
        "水库","隧道","特区","铁路","新村","雪峰","盐场","盐湖","渔场","直辖市",
        "自治区","自治县","自治州"};

    //Seperator type
    public static String SEPERATOR_C_SENTENCE = "。！？：；…";
    public static String SEPERATOR_C_SUB_SENTENCE = "、，（）“”‘’";
    public static String SEPERATOR_E_SENTENCE = "!?:;";
    public static String SEPERATOR_E_SUB_SENTENCE = ",()*'";
    //注释：原来程序为",()\042'"，"\042"为10进制42好ASC字符，为*
    public static String SEPERATOR_LINK = "\n\r 　";

    //Seperator between two words
    public static String WORD_SEGMENTER = "@";

    public static int MAX_SEGMENT_NUM = 10;

    public static final int MAX_FREQUENCY = 25146057; // 现在总词频25146057
    /**
     * Smoothing 平滑因子
     */
    public static final double dTemp = (double) 1 / MAX_FREQUENCY + 0.00001;
    /**
     * 平滑参数
     */
    public static final double dSmoothingPara = 0.1;
    /**
     * 地址 ns
     */
    public final static String TAG_PLACE = "未##地";
    /**
     * 句子的开始 begin
     */
    public final static String TAG_BIGIN = "始##始";
    /**
     * 其它
     */
    public final static String TAG_OTHER = "未##它";
    /**
     * 团体名词 nt
     */
    public final static String TAG_GROUP = "未##团";
    /**
     * 数词 m
     */
    public final static String TAG_NUMBER = "未##数";
    /**
     * 数量词 mq （现在觉得应该和数词同等处理，比如一个人和一人都是合理的）
     */
    public final static String TAG_QUANTIFIER = "未##量";
    /**
     * 专有名词 nx
     */
    public final static String TAG_PROPER = "未##专";
    /**
     * 时间 t
     */
    public final static String TAG_TIME = "未##时";
    /**
     * 字符串 x
     */
    public final static String TAG_CLUSTER = "未##串";
    /**
     * 结束 end
     */
    public final static String TAG_END = "末##末";
    /**
     * 人名 nr
     */
    public final static String TAG_PEOPLE = "未##人";

    /**
     * 日志组件
     */
    public static Logger logger = Logger.getLogger("HanLP");
    static
    {
        logger.setLevel(Level.WARNING);
    }

    /**
     * trie树文件后缀名
     */
    public final static String TRIE_EXT = ".trie.dat";
    /**
     * 值文件后缀名
     */
    public final static String VALUE_EXT = ".value.dat";

    /**
     * 逆转后缀名
     */
    public final static String REVERSE_EXT = ".reverse";

    /**
     * 二进制文件后缀
     */
    public final static String BIN_EXT = ".bin";
}
