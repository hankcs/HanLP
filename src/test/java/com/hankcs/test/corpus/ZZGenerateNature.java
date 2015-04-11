/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 3:35</create-date>
 *
 * <copyright file="__GenerateNature.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.tag.Nature;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class ZZGenerateNature extends TestCase
{
    public void testGenerate() throws Exception
    {
        String text = "n 名词\n" +
                "nr 人名\n" +
                "nrj 日语人名\n" +
                "nrf 音译人名\n" +
                "ns 地名\n" +
                "nsf 音译地名\n" +
                "nt 机构团体名\n" +
                "\t\tntc 公司名\n" +
                "\t\t\tntcf 工厂\n" +
                "\t\t\tntcb 银行\n" +
                "\t\t\tntch 酒店宾馆\n" +
                "\t\tnto 政府机构\n" +
                "\t\tntu 大学\n" +
                "\t\tnts 中小学\n" +
                "\t\tnth 医院\n" +
                "nh 医药疾病等健康相关名词\n" +
                "\t\tnhm 药品\n" +
                "\t\tnhd 疾病\n" +
                "nn 工作相关名词\n" +
                "nnt 职务职称\n" +
                "nnd 职业\n" +
                "ng 名词性语素\n" +
                "ni 机构相关（不是独立机构名）\n" +
                "\tnic 下属机构\n" +
                "\tnis 机构后缀\n" +
                "nm 物品名\n" +
                "\tnmc 化学品名\n" +
                "nb 生物名\n" +
                "\tnba 动物名\n" +
                "\tnbp 植物名\n" +
                "nz 其他专名\n" +
                "g 学术词汇\n" +
                "\tgm 数学相关词汇\n" +
                "\tgp 物理相关词汇\n" +
                "\tgc 化学相关词汇\n" +
                "\tgb 生物相关词汇\n" +
                "\t\tgbc 生物类别\n" +
                "\tgg 地理地质相关词汇\n" +
                "\tgi 计算机相关词汇\n" +
                "j 简称略语\n" +
                "i 成语\n" +
                "l 习用语\n" +
                "t 时间词\n" +
                "tg 时间词性语素\n" +
                "s 处所词\n" +
                "f 方位词\n" +
                "v 动词\n" +
                "vd 副动词\n" +
                "vn 名动词\n" +
                "vshi 动词“是”\n" +
                "vyou 动词“有”\n" +
                "vf 趋向动词\n" +
                "vx 形式动词\n" +
                "vi 不及物动词（内动词）\n" +
                "vl 动词性惯用语\n" +
                "vg 动词性语素\n" +
                "a 形容词\n" +
                "ad 副形词\n" +
                "an 名形词\n" +
                "ag 形容词性语素\n" +
                "al 形容词性惯用语\n" +
                "b 区别词\n" +
                "bl 区别词性惯用语\n" +
                "z 状态词\n" +
                "r 代词\n" +
                "rr 人称代词\n" +
                "rz 指示代词\n" +
                "rzt 时间指示代词\n" +
                "rzs 处所指示代词\n" +
                "rzv 谓词性指示代词\n" +
                "ry 疑问代词\n" +
                "ryt 时间疑问代词\n" +
                "rys 处所疑问代词\n" +
                "ryv 谓词性疑问代词\n" +
                "rg 代词性语素\n" +
                "m 数词\n" +
                "mq 数量词\n" +
                "q 量词\n" +
                "qv 动量词\n" +
                "qt 时量词\n" +
                "d 副词\n" +
                "p 介词\n" +
                "pba 介词“把”\n" +
                "pbei 介词“被”\n" +
                "c 连词\n" +
                "\tcc 并列连词\n" +
                "u 助词\n" +
                "uzhe 着\n" +
                "ule 了 喽\n" +
                "uguo 过\n" +
                "ude1 的 底\n" +
                "ude2 地\n" +
                "ude3 得\n" +
                "usuo 所\n" +
                "udeng 等 等等 云云\n" +
                "uyy 一样 一般 似的 般\n" +
                "udh 的话\n" +
                "uls 来讲 来说 而言 说来\n" +
                "\n" +
                "uzhi 之\n" +
                "ulian 连 （“连小学生都会”）\n" +
                "\n" +
                "e 叹词\n" +
                "y 语气词(delete yg)\n" +
                "o 拟声词\n" +
                "h 前缀\n" +
                "k 后缀\n" +
                "x 字符串\n" +
                "\txx 非语素字\n" +
                "\txu 网址URL\n" +
                "w 标点符号\n" +
                "wkz 左括号，全角：（ 〔  ［  ｛  《 【  〖 〈   半角：( [ { <\n" +
                "wky 右括号，全角：） 〕  ］ ｝ 》  】 〗 〉 半角： ) ] { >\n" +
                "wyz 左引号，全角：“ ‘ 『  \n" +
                "wyy 右引号，全角：” ’ 』 \n" +
                "wj 句号，全角：。\n" +
                "ww 问号，全角：？ 半角：?\n" +
                "wt 叹号，全角：！ 半角：!\n" +
                "wd 逗号，全角：， 半角：,\n" +
                "wf 分号，全角：； 半角： ;\n" +
                "wn 顿号，全角：、\n" +
                "wm 冒号，全角：： 半角： :\n" +
                "ws 省略号，全角：……  …\n" +
                "wp 破折号，全角：——   －－   ——－   半角：---  ----\n" +
                "wb 百分号千分号，全角：％ ‰   半角：%\n" +
                "wh 单位符号，全角：￥ ＄ ￡  °  ℃  半角：$\n" +
                "\n";
        String[] params = text.split("\\n");
        int i = 0;
        for (String p : params)
        {
            p = p.trim();
            if (p.length() == 0) continue;
            System.out.print(++i + " ");
            System.out.println(p);
//            int cut = p.indexOf(' ');
//            System.out.println("/**\n" +
//                                       "* " + p.substring(cut + 1) + "\n" +
//                                       "*/\n" +
//                                       p.substring(0, cut) +",\n");
        }
    }

    public void testSize() throws Exception
    {
            System.out.println(Nature.values().length);
    }
}
