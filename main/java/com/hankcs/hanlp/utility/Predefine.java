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

/**
 * @author He Han
 */
public class Predefine
{
    public final static double MIN_PROBABILITY = 1e-10;
    public final static int CT_SENTENCE_BEGIN = 1;        //Sentence begin
    public final static int CT_SENTENCE_END = 4;          //Sentence ending
    public final static int CT_SINGLE = 5;                //SINGLE byte
    public final static int CT_DELIMITER = CT_SINGLE + 1; //delimiter
    public final static int CT_CHINESE = CT_SINGLE + 2;   //Chinese Char
    public final static int CT_LETTER = CT_SINGLE + 3;    //HanYu Pinyin
    public final static int CT_NUM = CT_SINGLE + 4;       //HanYu Pinyin
    public final static int CT_INDEX = CT_SINGLE + 5;     //HanYu Pinyin
    public final static int CT_OTHER = CT_SINGLE + 12;    //Other

    public static String POSTFIX_SINGLE =
        "坝邦堡杯城池村单岛道堤店洞渡队法峰府冈港阁宫沟国海号河湖环集江奖礁角街井郡坑口矿里岭楼路门盟庙弄牌派坡铺旗桥区渠泉人山省市水寺塔台滩坛堂厅亭屯湾文屋溪峡县线乡巷型洋窑营屿语园苑院闸寨站镇州庄族陂庵町";

    public final static String[] POSTFIX_MUTIPLE = {"半岛","草原","城市","大堤","大公国","大桥","地区",
        "帝国","渡槽","港口","高速公路","高原","公路","公园","共和国","谷地","广场",
        "国道","海峡","胡同","机场","集镇","教区","街道","口岸","码头","煤矿",
        "牧场","农场","盆地","平原","丘陵","群岛","沙漠","沙洲","山脉","山丘",
        "水库","隧道","特区","铁路","新村","雪峰","盐场","盐湖","渔场","直辖市",
        "自治区","自治县","自治州",""};

    public static String TRANS_ENGLISH =
        "·—阿埃艾爱安昂敖奥澳笆芭巴白拜班邦保堡鲍北贝本比毕彼别波玻博勃伯泊卜布才采仓查差柴彻川茨慈次达大戴代丹旦但当道德得的登迪狄蒂帝丁东杜敦多额俄厄鄂恩尔伐法范菲芬费佛夫福弗甫噶盖干冈哥戈革葛格各根古瓜哈海罕翰汗汉豪合河赫亨侯呼胡华霍基吉及加贾坚简杰金京久居君喀卡凯坎康考柯科可克肯库奎拉喇莱来兰郎朗劳勒雷累楞黎理李里莉丽历利立力连廉良列烈林隆卢虏鲁路伦仑罗洛玛马买麦迈曼茅茂梅门蒙盟米蜜密敏明摩莫墨默姆木穆那娜纳乃奈南内尼年涅宁纽努诺欧帕潘畔庞培佩彭皮平泼普其契恰强乔切钦沁泉让热荣肉儒瑞若萨塞赛桑瑟森莎沙山善绍舍圣施诗石什史士守斯司丝苏素索塔泰坦汤唐陶特提汀图土吐托陀瓦万王旺威韦维魏温文翁沃乌吾武伍西锡希喜夏相香歇谢辛新牙雅亚彦尧叶依伊衣宜义因音英雍尤于约宰泽增詹珍治中仲朱诸卓孜祖佐伽娅尕腓滕济嘉津赖莲琳律略慕妮聂裴浦奇齐琴茹珊卫欣逊札哲智兹芙汶迦珀琪梵斐胥黛";
    public static String TRANS_RUSSIAN =
        "·阿安奥巴比彼波布察茨大德得丁杜尔法夫伏甫盖格哈基加坚捷金卡科可克库拉莱兰勒雷里历利连列卢鲁罗洛马梅蒙米姆娜涅宁诺帕泼普奇齐乔切日萨色山申什斯索塔坦特托娃维文乌西希谢亚耶叶依伊以扎佐柴达登蒂戈果海赫华霍吉季津柯理琳玛曼穆纳尼契钦丘桑沙舍泰图瓦万雅卓兹";
    public static String TRANS_JAPANESE =
        "安奥八白百邦保北倍本比滨博步部彩菜仓昌长朝池赤川船淳次村大代岛稻道德地典渡尔繁饭风福冈高工宫古谷关广桂贵好浩和合河黑横恒宏后户荒绘吉纪佳加见健江介金今进井静敬靖久酒菊俊康可克口梨理里礼栗丽利立凉良林玲铃柳隆鹿麻玛美萌弥敏木纳南男内鸟宁朋片平崎齐千前浅桥琴青清庆秋丘曲泉仁忍日荣若三森纱杉山善上伸神圣石实矢世市室水顺司松泰桃藤天田土万望尾未文武五舞西细夏宪相小孝新星行雄秀雅亚岩杨洋阳遥野也叶一伊衣逸义益樱永由有佑宇羽郁渊元垣原远月悦早造则泽增扎宅章昭沼真政枝知之植智治中忠仲竹助椎子佐阪坂堀荻菅薰浜濑鸠筱";

    //Translation type
    public static int TT_ENGLISH = 0;
    public static int TT_RUSSIAN = 1;
    public static int TT_JAPANESE = 2;

    //Seperator type
    public static String SEPERATOR_C_SENTENCE = "。！？：；…";
    public static String SEPERATOR_C_SUB_SENTENCE = "、，（）“”‘’";
    public static String SEPERATOR_E_SENTENCE = "!?:;";
    public static String SEPERATOR_E_SUB_SENTENCE = ",()*'";
    //注释：原来程序为",()\042'"，"\042"为10进制42好ASC字符，为*
    public static String SEPERATOR_LINK = "\n\r 　";

    //Sentence begin and ending String
    public static String SENTENCE_BEGIN = "始##始";
    public static String SENTENCE_END = "末##末";

    //Seperator between two words
    public static String WORD_SEGMENTER = "@";

    public static int CC_NUM = 6768;

    //The number of Chinese Char,including 5 empty position between 3756-3761
    public static int WORD_MAXLENGTH = 100;
    public static int WT_DELIMITER = 0;
    public static int WT_CHINESE = 1;
    public static int WT_OTHER = 2;

    public static int MAX_WORDS = 650;
    public static int MAX_SEGMENT_NUM = 10;

    public static final int MAX_FREQUENCY = 32910822 * 2; //2079997;   //7528283+329805  //1993123+86874  // 原词典1107668 // 新词典 32910822
    /**
     * Smoothing 平滑因子
     */
    public static final double dTemp = Math.sqrt((double) 1 / MAX_FREQUENCY);
    /**
     * 平滑参数
     */
    public static final double dSmoothingPara = 0.1;

    public static int MAX_SENTENCE_LEN = 100;

    //if MIN_PROBLEM==1 then INFINITE_VALUE 10000.00  //The shortest path
    //else INFINITE_VALUE 0.00  //infinite value
    public static int MIN_PROBLEM = 1;
    public static double INFINITE_VALUE = 10000.00;

    public static int MAX_WORDS_PER_SENTENCE = 120;
    public static int MAX_UNKNOWN_PER_SENTENCE = 200;
    public static int MAX_POS_PER_WORD = 20;
    public static int LITTLE_FREQUENCY = 6;

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
     * 人名
     */
    public final static String TAG_PEOPLE = "未##人";

}
