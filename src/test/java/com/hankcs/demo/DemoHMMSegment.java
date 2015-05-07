/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/7 19:01</create-date>
 *
 * <copyright file="DemoHMMSegment.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.HMM.HMMSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * 演示二阶隐马分词，这是一种基于字标注的分词方法，对未登录词支持较好，对已登录词的分词速度慢。综合性能不如CRF分词。
 * 还未稳定，请不要用于生产环境。二阶隐马标注分词效果尚且不好，许多开源分词器使用甚至使用一阶隐马（BiGram二元文法），
 * 效果可想而知。对基于字符的序列标注分词方法，我只推荐CRF。
 *
 * @author hankcs
 */
public class DemoHMMSegment
{
    public static void main(String[] args)
    {
        HanLP.Config.ShowTermNature = false;    // 关闭词性显示
        Segment segment = new HMMSegment();
        String[] sentenceArray = new String[]
                {
                        "HanLP是由一系列模型与算法组成的Java工具包，目标是普及自然语言处理在生产环境中的应用。",
                        "高锰酸钾，强氧化剂，紫红色晶体，可溶于水，遇乙醇即被还原。常用作消毒剂、水净化剂、氧化剂、漂白剂、毒气吸收剂、二氧化碳精制剂等。", // 专业名词有一定辨识能力
                        "《夜晚的骰子》通过描述浅草的舞女在暗夜中扔骰子的情景,寄托了作者对庶民生活区的情感",    // 非新闻语料
                        "这个像是真的[委屈]前面那个打扮太江户了，一点不上品...@hankcs",                       // 微博
                        "鼎泰丰的小笼一点味道也没有...每样都淡淡的...淡淡的，哪有食堂2A的好次",
                        "克里斯蒂娜·克罗尔说：不，我不是虎妈。我全家都热爱音乐，我也鼓励他们这么做。",
                        "今日APPS：Sago Mini Toolbox培养孩子动手能力",
                        "财政部副部长王保安调任国家统计局党组书记",
                        "2.34米男子娶1.53米女粉丝 称夫妻生活没问题",
                        "你看过穆赫兰道吗",
                        "乐视超级手机能否承载贾布斯的生态梦"
                };
        for (String sentence : sentenceArray)
        {
            List<Term> termList = segment.seg(sentence);
            System.out.println(termList);
        }

        // 测个速度
        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        System.out.println(segment.seg(text));
        long start = System.currentTimeMillis();
        int pressure = 1000;
        for (int i = 0; i < pressure; ++i)
        {
            segment.seg(text);
        }
        double costTime = (System.currentTimeMillis() - start) / (double)1000;
        System.out.printf("HMM2分词速度：%.2f字每秒\n", text.length() * pressure / costTime);
    }
}
