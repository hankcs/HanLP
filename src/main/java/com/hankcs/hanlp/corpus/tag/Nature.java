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

import com.hankcs.hanlp.corpus.util.CustomNatureUtility;

/**
 * 词性
 *
 * @author hankcs
 */
public enum Nature
{
    /**
     * 区别语素
     */
    bg,

    /**
     * 数语素
     */
    mg,

    /**
     * 名词性惯用语
     */
    nl,

    /**
     * 字母专名
     */
    nx,

    /**
     * 量词语素
     */
    qg,

    /**
     * 助词
     */
    ud,

    /**
     * 助词
     */
    uj,

    /**
     * 着
     */
    uz,

    /**
     * 过
     */
    ug,

    /**
     * 连词
     */
    ul,

    /**
     * 连词
     */
    uv,

    /**
     * 语气语素
     */
    yg,

    /**
     * 状态词
     */
    zg,

    // 以上标签来自ICT，以下标签来自北大

    /**
     * 名词
     */
    n,

    /**
     * 人名
     */
    nr,

    /**
     * 日语人名
     */
    nrj,

    /**
     * 音译人名
     */
    nrf,

    /**
     * 复姓
     */
    nr1,

    /**
     * 蒙古姓名
     */
    nr2,

    /**
     * 地名
     */
    ns,

    /**
     * 音译地名
     */
    nsf,

    /**
     * 机构团体名
     */
    nt,

    /**
     * 公司名
     */
    ntc,

    /**
     * 工厂
     */
    ntcf,

    /**
     * 银行
     */
    ntcb,

    /**
     * 酒店宾馆
     */
    ntch,

    /**
     * 政府机构
     */
    nto,

    /**
     * 大学
     */
    ntu,

    /**
     * 中小学
     */
    nts,

    /**
     * 医院
     */
    nth,

    /**
     * 医药疾病等健康相关名词
     */
    nh,

    /**
     * 药品
     */
    nhm,

    /**
     * 疾病
     */
    nhd,

    /**
     * 工作相关名词
     */
    nn,

    /**
     * 职务职称
     */
    nnt,

    /**
     * 职业
     */
    nnd,

    /**
     * 名词性语素
     */
    ng,

    /**
     * 食品，比如“薯片”
     */
    nf,

    /**
     * 机构相关（不是独立机构名）
     */
    ni,

    /**
     * 教育相关机构
     */
    nit,

    /**
     * 下属机构
     */
    nic,

    /**
     * 机构后缀
     */
    nis,

    /**
     * 物品名
     */
    nm,

    /**
     * 化学品名
     */
    nmc,

    /**
     * 生物名
     */
    nb,

    /**
     * 动物名
     */
    nba,

    /**
     * 动物纲目
     */
    nbc,

    /**
     * 植物名
     */
    nbp,

    /**
     * 其他专名
     */
    nz,

    /**
     * 学术词汇
     */
    g,

    /**
     * 数学相关词汇
     */
    gm,

    /**
     * 物理相关词汇
     */
    gp,

    /**
     * 化学相关词汇
     */
    gc,

    /**
     * 生物相关词汇
     */
    gb,

    /**
     * 生物类别
     */
    gbc,

    /**
     * 地理地质相关词汇
     */
    gg,

    /**
     * 计算机相关词汇
     */
    gi,

    /**
     * 简称略语
     */
    j,

    /**
     * 成语
     */
    i,

    /**
     * 习用语
     */
    l,

    /**
     * 时间词
     */
    t,

    /**
     * 时间词性语素
     */
    tg,

    /**
     * 处所词
     */
    s,

    /**
     * 方位词
     */
    f,

    /**
     * 动词
     */
    v,

    /**
     * 副动词
     */
    vd,

    /**
     * 名动词
     */
    vn,

    /**
     * 动词“是”
     */
    vshi,

    /**
     * 动词“有”
     */
    vyou,

    /**
     * 趋向动词
     */
    vf,

    /**
     * 形式动词
     */
    vx,

    /**
     * 不及物动词（内动词）
     */
    vi,

    /**
     * 动词性惯用语
     */
    vl,

    /**
     * 动词性语素
     */
    vg,

    /**
     * 形容词
     */
    a,

    /**
     * 副形词
     */
    ad,

    /**
     * 名形词
     */
    an,

    /**
     * 形容词性语素
     */
    ag,

    /**
     * 形容词性惯用语
     */
    al,

    /**
     * 区别词
     */
    b,

    /**
     * 区别词性惯用语
     */
    bl,

    /**
     * 状态词
     */
    z,

    /**
     * 代词
     */
    r,

    /**
     * 人称代词
     */
    rr,

    /**
     * 指示代词
     */
    rz,

    /**
     * 时间指示代词
     */
    rzt,

    /**
     * 处所指示代词
     */
    rzs,

    /**
     * 谓词性指示代词
     */
    rzv,

    /**
     * 疑问代词
     */
    ry,

    /**
     * 时间疑问代词
     */
    ryt,

    /**
     * 处所疑问代词
     */
    rys,

    /**
     * 谓词性疑问代词
     */
    ryv,

    /**
     * 代词性语素
     */
    rg,

    /**
     * 古汉语代词性语素
     */
    Rg,

    /**
     * 数词
     */
    m,

    /**
     * 数量词
     */
    mq,

    /**
     * 甲乙丙丁之类的数词
     */
    Mg,

    /**
     * 量词
     */
    q,

    /**
     * 动量词
     */
    qv,

    /**
     * 时量词
     */
    qt,

    /**
     * 副词
     */
    d,

    /**
     * 辄,俱,复之类的副词
     */
    dg,

    /**
     * 连语
     */
    dl,

    /**
     * 介词
     */
    p,

    /**
     * 介词“把”
     */
    pba,

    /**
     * 介词“被”
     */
    pbei,

    /**
     * 连词
     */
    c,

    /**
     * 并列连词
     */
    cc,

    /**
     * 助词
     */
    u,

    /**
     * 着
     */
    uzhe,

    /**
     * 了 喽
     */
    ule,

    /**
     * 过
     */
    uguo,

    /**
     * 的 底
     */
    ude1,

    /**
     * 地
     */
    ude2,

    /**
     * 得
     */
    ude3,

    /**
     * 所
     */
    usuo,

    /**
     * 等 等等 云云
     */
    udeng,

    /**
     * 一样 一般 似的 般
     */
    uyy,

    /**
     * 的话
     */
    udh,

    /**
     * 来讲 来说 而言 说来
     */
    uls,

    /**
     * 之
     */
    uzhi,

    /**
     * 连 （“连小学生都会”）
     */
    ulian,

    /**
     * 叹词
     */
    e,

    /**
     * 语气词(delete yg)
     */
    y,

    /**
     * 拟声词
     */
    o,

    /**
     * 前缀
     */
    h,

    /**
     * 后缀
     */
    k,

    /**
     * 字符串
     */
    x,

    /**
     * 非语素字
     */
    xx,

    /**
     * 网址URL
     */
    xu,

    /**
     * 标点符号
     */
    w,

    /**
     * 左括号，全角：（ 〔  ［  ｛  《 【  〖 〈   半角：( [ { <
     */
    wkz,

    /**
     * 右括号，全角：） 〕  ］ ｝ 》  】 〗 〉 半角： ) ] { >
     */
    wky,

    /**
     * 左引号，全角：“ ‘ 『
     */
    wyz,

    /**
     * 右引号，全角：” ’ 』
     */
    wyy,

    /**
     * 句号，全角：。
     */
    wj,

    /**
     * 问号，全角：？ 半角：?
     */
    ww,

    /**
     * 叹号，全角：！ 半角：!
     */
    wt,

    /**
     * 逗号，全角：， 半角：,
     */
    wd,

    /**
     * 分号，全角：； 半角： ;
     */
    wf,

    /**
     * 顿号，全角：、
     */
    wn,

    /**
     * 冒号，全角：： 半角： :
     */
    wm,

    /**
     * 省略号，全角：……  …
     */
    ws,

    /**
     * 破折号，全角：——   －－   ——－   半角：---  ----
     */
    wp,

    /**
     * 百分号千分号，全角：％ ‰   半角：%
     */
    wb,

    /**
     * 单位符号，全角：￥ ＄ ￡  °  ℃  半角：$
     */
    wh,

    /**
     * 仅用于终##终，不会出现在分词结果中
     */
    end,

    /**
     * 仅用于始##始，不会出现在分词结果中
     */
    begin,

    ;

    /**
     * 词性是否以该前缀开头<br>
     *     词性根据开头的几个字母可以判断大的类别
     * @param prefix 前缀
     * @return 是否以该前缀开头
     */
    public boolean startsWith(String prefix)
    {
        return toString().startsWith(prefix);
    }

    /**
     * 词性是否以该前缀开头<br>
     *     词性根据开头的几个字母可以判断大的类别
     * @param prefix 前缀
     * @return 是否以该前缀开头
     */
    public boolean startsWith(char prefix)
    {
        return toString().charAt(0) == prefix;
    }

    /**
     * 词性的首字母<br>
     *     词性根据开头的几个字母可以判断大的类别
     * @return
     */
    public char firstChar()
    {
        return toString().charAt(0);
    }

    /**
     * 安全地将字符串类型的词性转为Enum类型，如果未定义该词性，则返回null
     * @param name 字符串词性
     * @return Enum词性
     */
    public static Nature fromString(String name)
    {
        try
        {
            return Nature.valueOf(name);
        }
        catch (Exception e)
        {
            // 动态添加的词语有可能无法通过valueOf获取，所以遍历搜索
            for (Nature nature : Nature.values())
            {
                if (nature.toString().equals(name))
                {
                    return nature;
                }
            }
        }

        return null;
    }

    /**
     * 创建自定义词性,如果已有该对应词性,则直接返回已有的词性
     * @param name 字符串词性
     * @return Enum词性
     */
    public static Nature create(String name)
    {
        try
        {
            return Nature.valueOf(name);
        }
        catch (Exception e)
        {
            return CustomNatureUtility.addNature(name);
        }
    }
}