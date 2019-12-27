/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/10 14:34</create-date>
 *
 * <copyright file="Nature.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.tag;

import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 词性
 *
 * @author hankcs
 */
public class Nature
{
    /**
     * 区别语素
     */
    public static final Nature bg = new Nature("bg");

    /**
     * 数语素
     */
    public static final Nature mg = new Nature("mg");

    /**
     * 名词性惯用语
     */
    public static final Nature nl = new Nature("nl");

    /**
     * 字母专名
     */
    public static final Nature nx = new Nature("nx");

    /**
     * 量词语素
     */
    public static final Nature qg = new Nature("qg");

    /**
     * 助词
     */
    public static final Nature ud = new Nature("ud");

    /**
     * 助词
     */
    public static final Nature uj = new Nature("uj");

    /**
     * 着
     */
    public static final Nature uz = new Nature("uz");

    /**
     * 过
     */
    public static final Nature ug = new Nature("ug");

    /**
     * 连词
     */
    public static final Nature ul = new Nature("ul");

    /**
     * 连词
     */
    public static final Nature uv = new Nature("uv");

    /**
     * 语气语素
     */
    public static final Nature yg = new Nature("yg");

    /**
     * 状态词
     */
    public static final Nature zg = new Nature("zg");

    // 以上标签来自ICT，以下标签来自北大

    /**
     * 名词
     */
    public static final Nature n = new Nature("n");

    /**
     * 人名
     */
    public static final Nature nr = new Nature("nr");

    /**
     * 日语人名
     */
    public static final Nature nrj = new Nature("nrj");

    /**
     * 音译人名
     */
    public static final Nature nrf = new Nature("nrf");

    /**
     * 复姓
     */
    public static final Nature nr1 = new Nature("nr1");

    /**
     * 蒙古姓名
     */
    public static final Nature nr2 = new Nature("nr2");

    /**
     * 地名
     */
    public static final Nature ns = new Nature("ns");

    /**
     * 音译地名
     */
    public static final Nature nsf = new Nature("nsf");

    /**
     * 机构团体名
     */
    public static final Nature nt = new Nature("nt");

    /**
     * 公司名
     */
    public static final Nature ntc = new Nature("ntc");

    /**
     * 工厂
     */
    public static final Nature ntcf = new Nature("ntcf");

    /**
     * 银行
     */
    public static final Nature ntcb = new Nature("ntcb");

    /**
     * 酒店宾馆
     */
    public static final Nature ntch = new Nature("ntch");

    /**
     * 政府机构
     */
    public static final Nature nto = new Nature("nto");

    /**
     * 大学
     */
    public static final Nature ntu = new Nature("ntu");

    /**
     * 中小学
     */
    public static final Nature nts = new Nature("nts");

    /**
     * 医院
     */
    public static final Nature nth = new Nature("nth");

    /**
     * 医药疾病等健康相关名词
     */
    public static final Nature nh = new Nature("nh");

    /**
     * 药品
     */
    public static final Nature nhm = new Nature("nhm");

    /**
     * 疾病
     */
    public static final Nature nhd = new Nature("nhd");

    /**
     * 工作相关名词
     */
    public static final Nature nn = new Nature("nn");

    /**
     * 职务职称
     */
    public static final Nature nnt = new Nature("nnt");

    /**
     * 职业
     */
    public static final Nature nnd = new Nature("nnd");

    /**
     * 名词性语素
     */
    public static final Nature ng = new Nature("ng");

    /**
     * 食品，比如“薯片”
     */
    public static final Nature nf = new Nature("nf");

    /**
     * 机构相关（不是独立机构名）
     */
    public static final Nature ni = new Nature("ni");

    /**
     * 教育相关机构
     */
    public static final Nature nit = new Nature("nit");

    /**
     * 下属机构
     */
    public static final Nature nic = new Nature("nic");

    /**
     * 机构后缀
     */
    public static final Nature nis = new Nature("nis");

    /**
     * 物品名
     */
    public static final Nature nm = new Nature("nm");

    /**
     * 化学品名
     */
    public static final Nature nmc = new Nature("nmc");

    /**
     * 生物名
     */
    public static final Nature nb = new Nature("nb");

    /**
     * 动物名
     */
    public static final Nature nba = new Nature("nba");

    /**
     * 动物纲目
     */
    public static final Nature nbc = new Nature("nbc");

    /**
     * 植物名
     */
    public static final Nature nbp = new Nature("nbp");

    /**
     * 其他专名
     */
    public static final Nature nz = new Nature("nz");

    /**
     * 学术词汇
     */
    public static final Nature g = new Nature("g");

    /**
     * 数学相关词汇
     */
    public static final Nature gm = new Nature("gm");

    /**
     * 物理相关词汇
     */
    public static final Nature gp = new Nature("gp");

    /**
     * 化学相关词汇
     */
    public static final Nature gc = new Nature("gc");

    /**
     * 生物相关词汇
     */
    public static final Nature gb = new Nature("gb");

    /**
     * 生物类别
     */
    public static final Nature gbc = new Nature("gbc");

    /**
     * 地理地质相关词汇
     */
    public static final Nature gg = new Nature("gg");

    /**
     * 计算机相关词汇
     */
    public static final Nature gi = new Nature("gi");

    /**
     * 简称略语
     */
    public static final Nature j = new Nature("j");

    /**
     * 成语
     */
    public static final Nature i = new Nature("i");

    /**
     * 习用语
     */
    public static final Nature l = new Nature("l");

    /**
     * 时间词
     */
    public static final Nature t = new Nature("t");

    /**
     * 时间词性语素
     */
    public static final Nature tg = new Nature("tg");

    /**
     * 处所词
     */
    public static final Nature s = new Nature("s");

    /**
     * 方位词
     */
    public static final Nature f = new Nature("f");

    /**
     * 动词
     */
    public static final Nature v = new Nature("v");

    /**
     * 副动词
     */
    public static final Nature vd = new Nature("vd");

    /**
     * 名动词
     */
    public static final Nature vn = new Nature("vn");

    /**
     * 动词“是”
     */
    public static final Nature vshi = new Nature("vshi");

    /**
     * 动词“有”
     */
    public static final Nature vyou = new Nature("vyou");

    /**
     * 趋向动词
     */
    public static final Nature vf = new Nature("vf");

    /**
     * 形式动词
     */
    public static final Nature vx = new Nature("vx");

    /**
     * 不及物动词（内动词）
     */
    public static final Nature vi = new Nature("vi");

    /**
     * 动词性惯用语
     */
    public static final Nature vl = new Nature("vl");

    /**
     * 动词性语素
     */
    public static final Nature vg = new Nature("vg");

    /**
     * 形容词
     */
    public static final Nature a = new Nature("a");

    /**
     * 副形词
     */
    public static final Nature ad = new Nature("ad");

    /**
     * 名形词
     */
    public static final Nature an = new Nature("an");

    /**
     * 形容词性语素
     */
    public static final Nature ag = new Nature("ag");

    /**
     * 形容词性惯用语
     */
    public static final Nature al = new Nature("al");

    /**
     * 区别词
     */
    public static final Nature b = new Nature("b");

    /**
     * 区别词性惯用语
     */
    public static final Nature bl = new Nature("bl");

    /**
     * 状态词
     */
    public static final Nature z = new Nature("z");

    /**
     * 代词
     */
    public static final Nature r = new Nature("r");

    /**
     * 人称代词
     */
    public static final Nature rr = new Nature("rr");

    /**
     * 指示代词
     */
    public static final Nature rz = new Nature("rz");

    /**
     * 时间指示代词
     */
    public static final Nature rzt = new Nature("rzt");

    /**
     * 处所指示代词
     */
    public static final Nature rzs = new Nature("rzs");

    /**
     * 谓词性指示代词
     */
    public static final Nature rzv = new Nature("rzv");

    /**
     * 疑问代词
     */
    public static final Nature ry = new Nature("ry");

    /**
     * 时间疑问代词
     */
    public static final Nature ryt = new Nature("ryt");

    /**
     * 处所疑问代词
     */
    public static final Nature rys = new Nature("rys");

    /**
     * 谓词性疑问代词
     */
    public static final Nature ryv = new Nature("ryv");

    /**
     * 代词性语素
     */
    public static final Nature rg = new Nature("rg");

    /**
     * 古汉语代词性语素
     */
    public static final Nature Rg = new Nature("Rg");

    /**
     * 数词
     */
    public static final Nature m = new Nature("m");

    /**
     * 数量词
     */
    public static final Nature mq = new Nature("mq");

    /**
     * 甲乙丙丁之类的数词
     */
    public static final Nature Mg = new Nature("Mg");

    /**
     * 量词
     */
    public static final Nature q = new Nature("q");

    /**
     * 动量词
     */
    public static final Nature qv = new Nature("qv");

    /**
     * 时量词
     */
    public static final Nature qt = new Nature("qt");

    /**
     * 副词
     */
    public static final Nature d = new Nature("d");

    /**
     * 辄,俱,复之类的副词
     */
    public static final Nature dg = new Nature("dg");

    /**
     * 连语
     */
    public static final Nature dl = new Nature("dl");

    /**
     * 介词
     */
    public static final Nature p = new Nature("p");

    /**
     * 介词“把”
     */
    public static final Nature pba = new Nature("pba");

    /**
     * 介词“被”
     */
    public static final Nature pbei = new Nature("pbei");

    /**
     * 连词
     */
    public static final Nature c = new Nature("c");

    /**
     * 并列连词
     */
    public static final Nature cc = new Nature("cc");

    /**
     * 助词
     */
    public static final Nature u = new Nature("u");

    /**
     * 着
     */
    public static final Nature uzhe = new Nature("uzhe");

    /**
     * 了 喽
     */
    public static final Nature ule = new Nature("ule");

    /**
     * 过
     */
    public static final Nature uguo = new Nature("uguo");

    /**
     * 的 底
     */
    public static final Nature ude1 = new Nature("ude1");

    /**
     * 地
     */
    public static final Nature ude2 = new Nature("ude2");

    /**
     * 得
     */
    public static final Nature ude3 = new Nature("ude3");

    /**
     * 所
     */
    public static final Nature usuo = new Nature("usuo");

    /**
     * 等 等等 云云
     */
    public static final Nature udeng = new Nature("udeng");

    /**
     * 一样 一般 似的 般
     */
    public static final Nature uyy = new Nature("uyy");

    /**
     * 的话
     */
    public static final Nature udh = new Nature("udh");

    /**
     * 来讲 来说 而言 说来
     */
    public static final Nature uls = new Nature("uls");

    /**
     * 之
     */
    public static final Nature uzhi = new Nature("uzhi");

    /**
     * 连 （“连小学生都会”）
     */
    public static final Nature ulian = new Nature("ulian");

    /**
     * 叹词
     */
    public static final Nature e = new Nature("e");

    /**
     * 语气词(delete yg)
     */
    public static final Nature y = new Nature("y");

    /**
     * 拟声词
     */
    public static final Nature o = new Nature("o");

    /**
     * 前缀
     */
    public static final Nature h = new Nature("h");

    /**
     * 后缀
     */
    public static final Nature k = new Nature("k");

    /**
     * 字符串
     */
    public static final Nature x = new Nature("x");

    /**
     * 非语素字
     */
    public static final Nature xx = new Nature("xx");

    /**
     * 网址URL
     */
    public static final Nature xu = new Nature("xu");

    /**
     * 标点符号
     */
    public static final Nature w = new Nature("w");

    /**
     * 左括号，全角：（ 〔  ［  ｛  《 【  〖 〈   半角：( [ { <
     */
    public static final Nature wkz = new Nature("wkz");

    /**
     * 右括号，全角：） 〕  ］ ｝ 》  】 〗 〉 半角： ) ] { >
     */
    public static final Nature wky = new Nature("wky");

    /**
     * 左引号，全角：“ ‘ 『
     */
    public static final Nature wyz = new Nature("wyz");

    /**
     * 右引号，全角：” ’ 』
     */
    public static final Nature wyy = new Nature("wyy");

    /**
     * 句号，全角：。
     */
    public static final Nature wj = new Nature("wj");

    /**
     * 问号，全角：？ 半角：?
     */
    public static final Nature ww = new Nature("ww");

    /**
     * 叹号，全角：！ 半角：!
     */
    public static final Nature wt = new Nature("wt");

    /**
     * 逗号，全角：， 半角：,
     */
    public static final Nature wd = new Nature("wd");

    /**
     * 分号，全角：； 半角： ;
     */
    public static final Nature wf = new Nature("wf");

    /**
     * 顿号，全角：、
     */
    public static final Nature wn = new Nature("wn");

    /**
     * 冒号，全角：： 半角： :
     */
    public static final Nature wm = new Nature("wm");

    /**
     * 省略号，全角：……  …
     */
    public static final Nature ws = new Nature("ws");

    /**
     * 破折号，全角：——   －－   ——－   半角：---  ----
     */
    public static final Nature wp = new Nature("wp");

    /**
     * 百分号千分号，全角：％ ‰   半角：%
     */
    public static final Nature wb = new Nature("wb");

    /**
     * 单位符号，全角：￥ ＄ ￡  °  ℃  半角：$
     */
    public static final Nature wh = new Nature("wh");

    /**
     * 仅用于终##终，不会出现在分词结果中
     */
    public static final Nature end = new Nature("end");

    /**
     * 仅用于始##始，不会出现在分词结果中
     */
    public static final Nature begin = new Nature("begin");

    private static ConcurrentHashMap<String, Integer> idMap;
    private static Nature[] values;
    private int ordinal;
    private final String name;

    private Nature(String name)
    {
        if (idMap == null) {
            idMap = new ConcurrentHashMap<String, Integer>();
        }
        assert !idMap.containsKey(name);
        this.name = name;
        ordinal = idMap.size();
        idMap.put(name, ordinal);
        Nature[] extended = new Nature[idMap.size()];
        if (values != null){
            System.arraycopy(values, 0, extended, 0, values.length);
        }
        extended[ordinal] = this;
        values = extended;
    }

    /**
     * 词性是否以该前缀开头<br>
     * 词性根据开头的几个字母可以判断大的类别
     *
     * @param prefix 前缀
     * @return 是否以该前缀开头
     */
    public boolean startsWith(String prefix)
    {
        return name.startsWith(prefix);
    }

    /**
     * 词性是否以该前缀开头<br>
     * 词性根据开头的几个字母可以判断大的类别
     *
     * @param prefix 前缀
     * @return 是否以该前缀开头
     */
    public boolean startsWith(char prefix)
    {
        return name.charAt(0) == prefix;
    }

    /**
     * 词性的首字母<br>
     * 词性根据开头的几个字母可以判断大的类别
     *
     * @return
     */
    public char firstChar()
    {
        return name.charAt(0);
    }

    /**
     * 安全地将字符串类型的词性转为Enum类型，如果未定义该词性，则返回null
     *
     * @param name 字符串词性
     * @return Enum词性
     */
    public static final Nature fromString(String name)
    {
        Integer id = idMap.get(name);
        if (id == null)
            return null;
        return values[id];
    }

    /**
     * 创建自定义词性,如果已有该对应词性,则直接返回已有的词性
     *
     * @param name 字符串词性
     * @return Enum词性
     */
    public static final Nature create(String name)
    {
        Nature nature = fromString(name);
        if (nature == null)
            return new Nature(name);
        return nature;
    }

    @Override
    public String toString()
    {
        return name;
    }

    public int ordinal()
    {
        return ordinal;
    }

    public static Nature[] values()
    {
        return values;
    }
}
