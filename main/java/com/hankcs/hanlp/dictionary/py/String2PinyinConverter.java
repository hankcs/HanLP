/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/5 21:17</create-date>
 *
 * <copyright file="PinyinGuesser.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.py;

import com.hankcs.hanlp.algoritm.ahocorasick.trie.Token;
import com.hankcs.hanlp.algoritm.ahocorasick.trie.Trie;
import javafx.util.Pair;

import java.util.*;

/**
 * 汉字转拼音，提供拼音字符串转拼音，支持汉英混合的杂乱文本
 * @author hankcs
 */
public class String2PinyinConverter
{
    static Trie trie;
    static Map<String, Pinyin> map;
    /**
     * 将音调统一换为1，下标为拼音的ordinal，值为音调1
     */
    static Pinyin[] tone2tone1 = new Pinyin[]{Pinyin.a1,Pinyin.a1,Pinyin.a1,Pinyin.a1,Pinyin.a1,Pinyin.ai1,Pinyin.ai1,Pinyin.ai1,Pinyin.ai1,Pinyin.an1,Pinyin.an1,Pinyin.an1,Pinyin.an1,Pinyin.ang1,Pinyin.ang1,Pinyin.ang1,Pinyin.ang1,Pinyin.ao1,Pinyin.ao1,Pinyin.ao1,Pinyin.ao1,Pinyin.ba1,Pinyin.ba1,Pinyin.ba1,Pinyin.ba1,Pinyin.ba1,Pinyin.bai1,Pinyin.bai1,Pinyin.bai1,Pinyin.bai1,Pinyin.ban1,Pinyin.ban1,Pinyin.ban1,Pinyin.bang1,Pinyin.bang1,Pinyin.bang1,Pinyin.bao1,Pinyin.bao1,Pinyin.bao1,Pinyin.bao1,Pinyin.bei1,Pinyin.bei1,Pinyin.bei1,Pinyin.bei1,Pinyin.ben1,Pinyin.ben1,Pinyin.ben1,Pinyin.beng1,Pinyin.beng1,Pinyin.beng1,Pinyin.beng1,Pinyin.bi1,Pinyin.bi1,Pinyin.bi1,Pinyin.bi1,Pinyin.bian1,Pinyin.bian1,Pinyin.bian1,Pinyin.bian1,Pinyin.biao1,Pinyin.biao1,Pinyin.biao1,Pinyin.biao1,Pinyin.bie1,Pinyin.bie1,Pinyin.bie1,Pinyin.bie1,Pinyin.bin1,Pinyin.bin1,Pinyin.bin1,Pinyin.bing1,Pinyin.bing1,Pinyin.bing1,Pinyin.bo1,Pinyin.bo1,Pinyin.bo1,Pinyin.bo1,Pinyin.bo1,Pinyin.bu1,Pinyin.bu1,Pinyin.bu1,Pinyin.bu1,Pinyin.ca1,Pinyin.ca1,Pinyin.ca1,Pinyin.cai1,Pinyin.cai1,Pinyin.cai1,Pinyin.cai1,Pinyin.can1,Pinyin.can1,Pinyin.can1,Pinyin.can1,Pinyin.cang1,Pinyin.cang1,Pinyin.cang1,Pinyin.cang1,Pinyin.cao1,Pinyin.cao1,Pinyin.cao1,Pinyin.cao1,Pinyin.ce4,Pinyin.cen1,Pinyin.cen1,Pinyin.ceng1,Pinyin.ceng1,Pinyin.ceng1,Pinyin.cha1,Pinyin.cha1,Pinyin.cha1,Pinyin.cha1,Pinyin.cha1,Pinyin.chai1,Pinyin.chai1,Pinyin.chai1,Pinyin.chai1,Pinyin.chan1,Pinyin.chan1,Pinyin.chan1,Pinyin.chan1,Pinyin.chang1,Pinyin.chang1,Pinyin.chang1,Pinyin.chang1,Pinyin.chang1,Pinyin.chao1,Pinyin.chao1,Pinyin.chao1,Pinyin.chao1,Pinyin.che1,Pinyin.che1,Pinyin.che1,Pinyin.chen1,Pinyin.chen1,Pinyin.chen1,Pinyin.chen1,Pinyin.chen1,Pinyin.cheng1,Pinyin.cheng1,Pinyin.cheng1,Pinyin.cheng1,Pinyin.chi1,Pinyin.chi1,Pinyin.chi1,Pinyin.chi1,Pinyin.chi1,Pinyin.chong1,Pinyin.chong1,Pinyin.chong1,Pinyin.chong1,Pinyin.chou1,Pinyin.chou1,Pinyin.chou1,Pinyin.chou1,Pinyin.chou1,Pinyin.chu1,Pinyin.chu1,Pinyin.chu1,Pinyin.chu1,Pinyin.chu1,Pinyin.chua1,Pinyin.chuai1,Pinyin.chuai1,Pinyin.chuai1,Pinyin.chuai1,Pinyin.chuan1,Pinyin.chuan1,Pinyin.chuan1,Pinyin.chuan1,Pinyin.chuang1,Pinyin.chuang1,Pinyin.chuang1,Pinyin.chuang1,Pinyin.chui1,Pinyin.chui1,Pinyin.chui1,Pinyin.chun1,Pinyin.chun1,Pinyin.chun1,Pinyin.chuo1,Pinyin.chuo1,Pinyin.chuo1,Pinyin.chuo1,Pinyin.ci1,Pinyin.ci1,Pinyin.ci1,Pinyin.ci1,Pinyin.cong1,Pinyin.cong1,Pinyin.cong1,Pinyin.cou3,Pinyin.cou3,Pinyin.cu1,Pinyin.cu1,Pinyin.cu1,Pinyin.cu1,Pinyin.cuan1,Pinyin.cuan1,Pinyin.cuan1,Pinyin.cui1,Pinyin.cui1,Pinyin.cui1,Pinyin.cui1,Pinyin.cun1,Pinyin.cun1,Pinyin.cun1,Pinyin.cun1,Pinyin.cuo1,Pinyin.cuo1,Pinyin.cuo1,Pinyin.cuo1,Pinyin.da1,Pinyin.da1,Pinyin.da1,Pinyin.da1,Pinyin.da1,Pinyin.dai1,Pinyin.dai1,Pinyin.dai1,Pinyin.dan1,Pinyin.dan1,Pinyin.dan1,Pinyin.dan1,Pinyin.dang1,Pinyin.dang1,Pinyin.dang1,Pinyin.dao1,Pinyin.dao1,Pinyin.dao1,Pinyin.dao1,Pinyin.de1,Pinyin.de1,Pinyin.de1,Pinyin.dei1,Pinyin.dei1,Pinyin.den1,Pinyin.den1,Pinyin.deng1,Pinyin.deng1,Pinyin.deng1,Pinyin.di1,Pinyin.di1,Pinyin.di1,Pinyin.di1,Pinyin.dia3,Pinyin.dian1,Pinyin.dian1,Pinyin.dian1,Pinyin.diao1,Pinyin.diao1,Pinyin.diao1,Pinyin.die1,Pinyin.die1,Pinyin.die1,Pinyin.ding1,Pinyin.ding1,Pinyin.ding1,Pinyin.ding1,Pinyin.diu1,Pinyin.dong1,Pinyin.dong1,Pinyin.dong1,Pinyin.dou1,Pinyin.dou1,Pinyin.dou1,Pinyin.dou1,Pinyin.du1,Pinyin.du1,Pinyin.du1,Pinyin.du1,Pinyin.duan1,Pinyin.duan1,Pinyin.duan1,Pinyin.dui1,Pinyin.dui1,Pinyin.dui1,Pinyin.dun1,Pinyin.dun1,Pinyin.dun1,Pinyin.dun1,Pinyin.duo1,Pinyin.duo1,Pinyin.duo1,Pinyin.duo1,Pinyin.duo1,Pinyin.e1,Pinyin.e1,Pinyin.e1,Pinyin.e1,Pinyin.ei1,Pinyin.ei1,Pinyin.ei1,Pinyin.ei1,Pinyin.en1,Pinyin.en1,Pinyin.en1,Pinyin.eng1,Pinyin.er2,Pinyin.er2,Pinyin.er2,Pinyin.er2,Pinyin.fa1,Pinyin.fa1,Pinyin.fa1,Pinyin.fa1,Pinyin.fan1,Pinyin.fan1,Pinyin.fan1,Pinyin.fan1,Pinyin.fang1,Pinyin.fang1,Pinyin.fang1,Pinyin.fang1,Pinyin.fang1,Pinyin.fei1,Pinyin.fei1,Pinyin.fei1,Pinyin.fei1,Pinyin.fen1,Pinyin.fen1,Pinyin.fen1,Pinyin.fen1,Pinyin.feng1,Pinyin.feng1,Pinyin.feng1,Pinyin.feng1,Pinyin.fiao4,Pinyin.fo2,Pinyin.fou1,Pinyin.fou1,Pinyin.fou1,Pinyin.fou1,Pinyin.fu1,Pinyin.fu1,Pinyin.fu1,Pinyin.fu1,Pinyin.fu1,Pinyin.ga1,Pinyin.ga1,Pinyin.ga1,Pinyin.ga1,Pinyin.gai1,Pinyin.gai1,Pinyin.gai1,Pinyin.gan1,Pinyin.gan1,Pinyin.gan1,Pinyin.gan1,Pinyin.gang1,Pinyin.gang1,Pinyin.gang1,Pinyin.gao1,Pinyin.gao1,Pinyin.gao1,Pinyin.ge1,Pinyin.ge1,Pinyin.ge1,Pinyin.ge1,Pinyin.gei3,Pinyin.gen1,Pinyin.gen1,Pinyin.gen1,Pinyin.gen1,Pinyin.geng1,Pinyin.geng1,Pinyin.geng1,Pinyin.gong1,Pinyin.gong1,Pinyin.gong1,Pinyin.gou1,Pinyin.gou1,Pinyin.gou1,Pinyin.gu1,Pinyin.gu1,Pinyin.gu1,Pinyin.gu1,Pinyin.gua1,Pinyin.gua1,Pinyin.gua1,Pinyin.guai1,Pinyin.guai1,Pinyin.guai1,Pinyin.guai1,Pinyin.guan1,Pinyin.guan1,Pinyin.guan1,Pinyin.guang1,Pinyin.guang1,Pinyin.guang1,Pinyin.gui1,Pinyin.gui1,Pinyin.gui1,Pinyin.gui1,Pinyin.gun1,Pinyin.gun1,Pinyin.gun1,Pinyin.guo1,Pinyin.guo1,Pinyin.guo1,Pinyin.guo1,Pinyin.guo1,Pinyin.ha1,Pinyin.ha1,Pinyin.ha1,Pinyin.ha1,Pinyin.hai1,Pinyin.hai1,Pinyin.hai1,Pinyin.hai1,Pinyin.han1,Pinyin.han1,Pinyin.han1,Pinyin.han1,Pinyin.han1,Pinyin.hang1,Pinyin.hang1,Pinyin.hang1,Pinyin.hang1,Pinyin.hao1,Pinyin.hao1,Pinyin.hao1,Pinyin.hao1,Pinyin.he1,Pinyin.he1,Pinyin.he1,Pinyin.hei1,Pinyin.hen1,Pinyin.hen1,Pinyin.hen1,Pinyin.hen1,Pinyin.heng1,Pinyin.heng1,Pinyin.heng1,Pinyin.hong1,Pinyin.hong1,Pinyin.hong1,Pinyin.hong1,Pinyin.hou1,Pinyin.hou1,Pinyin.hou1,Pinyin.hou1,Pinyin.hu1,Pinyin.hu1,Pinyin.hu1,Pinyin.hu1,Pinyin.hua1,Pinyin.hua1,Pinyin.hua1,Pinyin.hua1,Pinyin.huai1,Pinyin.huai1,Pinyin.huai1,Pinyin.huan1,Pinyin.huan1,Pinyin.huan1,Pinyin.huan1,Pinyin.huang1,Pinyin.huang1,Pinyin.huang1,Pinyin.huang1,Pinyin.hui1,Pinyin.hui1,Pinyin.hui1,Pinyin.hui1,Pinyin.hun1,Pinyin.hun1,Pinyin.hun1,Pinyin.hun1,Pinyin.huo1,Pinyin.huo1,Pinyin.huo1,Pinyin.huo1,Pinyin.huo1,Pinyin.ja4,Pinyin.ji1,Pinyin.ji1,Pinyin.ji1,Pinyin.ji1,Pinyin.ji1,Pinyin.jia1,Pinyin.jia1,Pinyin.jia1,Pinyin.jia1,Pinyin.jia1,Pinyin.jian1,Pinyin.jian1,Pinyin.jian1,Pinyin.jiang1,Pinyin.jiang1,Pinyin.jiang1,Pinyin.jiao1,Pinyin.jiao1,Pinyin.jiao1,Pinyin.jiao1,Pinyin.jie1,Pinyin.jie1,Pinyin.jie1,Pinyin.jie1,Pinyin.jie1,Pinyin.jin1,Pinyin.jin1,Pinyin.jin1,Pinyin.jing1,Pinyin.jing1,Pinyin.jing1,Pinyin.jiong1,Pinyin.jiong1,Pinyin.jiong1,Pinyin.jiu1,Pinyin.jiu1,Pinyin.jiu1,Pinyin.ju1,Pinyin.ju1,Pinyin.ju1,Pinyin.ju1,Pinyin.ju1,Pinyin.juan1,Pinyin.juan1,Pinyin.juan1,Pinyin.juan1,Pinyin.jue1,Pinyin.jue1,Pinyin.jue1,Pinyin.jue1,Pinyin.jun1,Pinyin.jun1,Pinyin.jun1,Pinyin.ka1,Pinyin.ka1,Pinyin.ka1,Pinyin.kai1,Pinyin.kai1,Pinyin.kai1,Pinyin.kan1,Pinyin.kan1,Pinyin.kan1,Pinyin.kang1,Pinyin.kang1,Pinyin.kang1,Pinyin.kang1,Pinyin.kao1,Pinyin.kao1,Pinyin.kao1,Pinyin.kao1,Pinyin.ke1,Pinyin.ke1,Pinyin.ke1,Pinyin.ke1,Pinyin.ke1,Pinyin.kei1,Pinyin.ken3,Pinyin.ken3,Pinyin.keng1,Pinyin.keng1,Pinyin.kong1,Pinyin.kong1,Pinyin.kong1,Pinyin.kou1,Pinyin.kou1,Pinyin.kou1,Pinyin.ku1,Pinyin.ku1,Pinyin.ku1,Pinyin.kua1,Pinyin.kua1,Pinyin.kua1,Pinyin.kuai1,Pinyin.kuai1,Pinyin.kuai1,Pinyin.kuan1,Pinyin.kuan1,Pinyin.kuang1,Pinyin.kuang1,Pinyin.kuang1,Pinyin.kuang1,Pinyin.kui1,Pinyin.kui1,Pinyin.kui1,Pinyin.kui1,Pinyin.kun1,Pinyin.kun1,Pinyin.kun1,Pinyin.kuo3,Pinyin.kuo3,Pinyin.la1,Pinyin.la1,Pinyin.la1,Pinyin.la1,Pinyin.la1,Pinyin.lai2,Pinyin.lai2,Pinyin.lai2,Pinyin.lan2,Pinyin.lan2,Pinyin.lan2,Pinyin.lan2,Pinyin.lang1,Pinyin.lang1,Pinyin.lang1,Pinyin.lang1,Pinyin.lao1,Pinyin.lao1,Pinyin.lao1,Pinyin.lao1,Pinyin.le1,Pinyin.le1,Pinyin.le1,Pinyin.lei1,Pinyin.lei1,Pinyin.lei1,Pinyin.lei1,Pinyin.lei1,Pinyin.leng1,Pinyin.leng1,Pinyin.leng1,Pinyin.leng1,Pinyin.li1,Pinyin.li1,Pinyin.li1,Pinyin.li1,Pinyin.li1,Pinyin.lia3,Pinyin.lian1,Pinyin.lian1,Pinyin.lian1,Pinyin.lian1,Pinyin.liang2,Pinyin.liang2,Pinyin.liang2,Pinyin.liang2,Pinyin.liao1,Pinyin.liao1,Pinyin.liao1,Pinyin.liao1,Pinyin.lie1,Pinyin.lie1,Pinyin.lie1,Pinyin.lie1,Pinyin.lie1,Pinyin.lin1,Pinyin.lin1,Pinyin.lin1,Pinyin.lin1,Pinyin.ling1,Pinyin.ling1,Pinyin.ling1,Pinyin.ling1,Pinyin.liu1,Pinyin.liu1,Pinyin.liu1,Pinyin.liu1,Pinyin.lo5,Pinyin.long1,Pinyin.long1,Pinyin.long1,Pinyin.long1,Pinyin.lou1,Pinyin.lou1,Pinyin.lou1,Pinyin.lou1,Pinyin.lou1,Pinyin.lu1,Pinyin.lu1,Pinyin.lu1,Pinyin.lu1,Pinyin.lu1,Pinyin.luan2,Pinyin.luan2,Pinyin.luan2,Pinyin.lun1,Pinyin.lun1,Pinyin.lun1,Pinyin.lun1,Pinyin.luo1,Pinyin.luo1,Pinyin.luo1,Pinyin.luo1,Pinyin.luo1,Pinyin.lv2,Pinyin.lv2,Pinyin.lv2,Pinyin.lve3,Pinyin.lve3,Pinyin.ma1,Pinyin.ma1,Pinyin.ma1,Pinyin.ma1,Pinyin.ma1,Pinyin.mai2,Pinyin.mai2,Pinyin.mai2,Pinyin.man1,Pinyin.man1,Pinyin.man1,Pinyin.man1,Pinyin.mang1,Pinyin.mang1,Pinyin.mang1,Pinyin.mao1,Pinyin.mao1,Pinyin.mao1,Pinyin.mao1,Pinyin.me1,Pinyin.me1,Pinyin.me1,Pinyin.mei2,Pinyin.mei2,Pinyin.mei2,Pinyin.men1,Pinyin.men1,Pinyin.men1,Pinyin.men1,Pinyin.men1,Pinyin.meng1,Pinyin.meng1,Pinyin.meng1,Pinyin.meng1,Pinyin.mi1,Pinyin.mi1,Pinyin.mi1,Pinyin.mi1,Pinyin.mian2,Pinyin.mian2,Pinyin.mian2,Pinyin.miao1,Pinyin.miao1,Pinyin.miao1,Pinyin.miao1,Pinyin.mie1,Pinyin.mie1,Pinyin.min2,Pinyin.min2,Pinyin.ming2,Pinyin.ming2,Pinyin.ming2,Pinyin.miu4,Pinyin.mo1,Pinyin.mo1,Pinyin.mo1,Pinyin.mo1,Pinyin.mo1,Pinyin.mou1,Pinyin.mou1,Pinyin.mou1,Pinyin.mou1,Pinyin.mu2,Pinyin.mu2,Pinyin.mu2,Pinyin.na1,Pinyin.na1,Pinyin.na1,Pinyin.na1,Pinyin.na1,Pinyin.nai2,Pinyin.nai2,Pinyin.nai2,Pinyin.nan1,Pinyin.nan1,Pinyin.nan1,Pinyin.nan1,Pinyin.nang1,Pinyin.nang1,Pinyin.nang1,Pinyin.nang1,Pinyin.nao1,Pinyin.nao1,Pinyin.nao1,Pinyin.nao1,Pinyin.ne2,Pinyin.ne2,Pinyin.ne2,Pinyin.nei3,Pinyin.nei3,Pinyin.nen1,Pinyin.nen1,Pinyin.nen1,Pinyin.neng2,Pinyin.neng2,Pinyin.ni1,Pinyin.ni1,Pinyin.ni1,Pinyin.ni1,Pinyin.nian1,Pinyin.nian1,Pinyin.nian1,Pinyin.nian1,Pinyin.niang2,Pinyin.niang2,Pinyin.niao3,Pinyin.niao3,Pinyin.nie1,Pinyin.nie1,Pinyin.nie1,Pinyin.nie1,Pinyin.nin2,Pinyin.nin2,Pinyin.ning2,Pinyin.ning2,Pinyin.ning2,Pinyin.niu1,Pinyin.niu1,Pinyin.niu1,Pinyin.niu1,Pinyin.nong2,Pinyin.nong2,Pinyin.nong2,Pinyin.nou2,Pinyin.nou2,Pinyin.nu2,Pinyin.nu2,Pinyin.nu2,Pinyin.nuan2,Pinyin.nuan2,Pinyin.nuan2,Pinyin.nun2,Pinyin.nun2,Pinyin.nuo2,Pinyin.nuo2,Pinyin.nuo2,Pinyin.nv3,Pinyin.nv3,Pinyin.nve4,Pinyin.o1,Pinyin.o1,Pinyin.o1,Pinyin.o1,Pinyin.o1,Pinyin.ou1,Pinyin.ou1,Pinyin.ou1,Pinyin.ou1,Pinyin.ou1,Pinyin.pa1,Pinyin.pa1,Pinyin.pa1,Pinyin.pai1,Pinyin.pai1,Pinyin.pai1,Pinyin.pai1,Pinyin.pan1,Pinyin.pan1,Pinyin.pan1,Pinyin.pan1,Pinyin.pang1,Pinyin.pang1,Pinyin.pang1,Pinyin.pang1,Pinyin.pang1,Pinyin.pao1,Pinyin.pao1,Pinyin.pao1,Pinyin.pao1,Pinyin.pei1,Pinyin.pei1,Pinyin.pei1,Pinyin.pei1,Pinyin.pen1,Pinyin.pen1,Pinyin.pen1,Pinyin.pen1,Pinyin.pen1,Pinyin.peng1,Pinyin.peng1,Pinyin.peng1,Pinyin.peng1,Pinyin.pi1,Pinyin.pi1,Pinyin.pi1,Pinyin.pi1,Pinyin.pi1,Pinyin.pian1,Pinyin.pian1,Pinyin.pian1,Pinyin.pian1,Pinyin.piao1,Pinyin.piao1,Pinyin.piao1,Pinyin.piao1,Pinyin.pie1,Pinyin.pie1,Pinyin.pie1,Pinyin.pin1,Pinyin.pin1,Pinyin.pin1,Pinyin.pin1,Pinyin.ping1,Pinyin.ping1,Pinyin.ping1,Pinyin.ping1,Pinyin.po1,Pinyin.po1,Pinyin.po1,Pinyin.po1,Pinyin.po1,Pinyin.pou1,Pinyin.pou1,Pinyin.pou1,Pinyin.pou1,Pinyin.pu1,Pinyin.pu1,Pinyin.pu1,Pinyin.pu1,Pinyin.qi1,Pinyin.qi1,Pinyin.qi1,Pinyin.qi1,Pinyin.qi1,Pinyin.qia1,Pinyin.qia1,Pinyin.qia1,Pinyin.qian1,Pinyin.qian1,Pinyin.qian1,Pinyin.qian1,Pinyin.qian1,Pinyin.qiang1,Pinyin.qiang1,Pinyin.qiang1,Pinyin.qiang1,Pinyin.qiao1,Pinyin.qiao1,Pinyin.qiao1,Pinyin.qiao1,Pinyin.qie1,Pinyin.qie1,Pinyin.qie1,Pinyin.qie1,Pinyin.qin1,Pinyin.qin1,Pinyin.qin1,Pinyin.qin1,Pinyin.qing1,Pinyin.qing1,Pinyin.qing1,Pinyin.qing1,Pinyin.qiong1,Pinyin.qiong1,Pinyin.qiong1,Pinyin.qiu1,Pinyin.qiu1,Pinyin.qiu1,Pinyin.qiu1,Pinyin.qu1,Pinyin.qu1,Pinyin.qu1,Pinyin.qu1,Pinyin.quan1,Pinyin.quan1,Pinyin.quan1,Pinyin.quan1,Pinyin.que1,Pinyin.que1,Pinyin.que1,Pinyin.qun1,Pinyin.qun1,Pinyin.qun1,Pinyin.ran2,Pinyin.ran2,Pinyin.rang1,Pinyin.rang1,Pinyin.rang1,Pinyin.rang1,Pinyin.rao2,Pinyin.rao2,Pinyin.rao2,Pinyin.re2,Pinyin.re2,Pinyin.re2,Pinyin.ren2,Pinyin.ren2,Pinyin.ren2,Pinyin.reng1,Pinyin.reng1,Pinyin.reng1,Pinyin.ri4,Pinyin.rong2,Pinyin.rong2,Pinyin.rong2,Pinyin.rou2,Pinyin.rou2,Pinyin.rou2,Pinyin.ru1,Pinyin.ru1,Pinyin.ru1,Pinyin.ru1,Pinyin.ruan2,Pinyin.ruan2,Pinyin.ruan2,Pinyin.rui2,Pinyin.rui2,Pinyin.rui2,Pinyin.run2,Pinyin.run2,Pinyin.ruo2,Pinyin.ruo2,Pinyin.sa1,Pinyin.sa1,Pinyin.sa1,Pinyin.sai1,Pinyin.sai1,Pinyin.sai1,Pinyin.sai1,Pinyin.san1,Pinyin.san1,Pinyin.san1,Pinyin.san1,Pinyin.sang1,Pinyin.sang1,Pinyin.sang1,Pinyin.sang1,Pinyin.sao1,Pinyin.sao1,Pinyin.sao1,Pinyin.se1,Pinyin.se1,Pinyin.sen1,Pinyin.sen1,Pinyin.seng1,Pinyin.sha1,Pinyin.sha1,Pinyin.sha1,Pinyin.sha1,Pinyin.shai1,Pinyin.shai1,Pinyin.shai1,Pinyin.shan1,Pinyin.shan1,Pinyin.shan1,Pinyin.shan1,Pinyin.shang1,Pinyin.shang1,Pinyin.shang1,Pinyin.shang1,Pinyin.shang1,Pinyin.shao1,Pinyin.shao1,Pinyin.shao1,Pinyin.shao1,Pinyin.she1,Pinyin.she1,Pinyin.she1,Pinyin.she1,Pinyin.shei2,Pinyin.shen1,Pinyin.shen1,Pinyin.shen1,Pinyin.shen1,Pinyin.sheng1,Pinyin.sheng1,Pinyin.sheng1,Pinyin.sheng1,Pinyin.shi1,Pinyin.shi1,Pinyin.shi1,Pinyin.shi1,Pinyin.shi1,Pinyin.shou1,Pinyin.shou1,Pinyin.shou1,Pinyin.shou1,Pinyin.shu1,Pinyin.shu1,Pinyin.shu1,Pinyin.shu1,Pinyin.shua1,Pinyin.shua1,Pinyin.shua1,Pinyin.shuai1,Pinyin.shuai1,Pinyin.shuai1,Pinyin.shuan1,Pinyin.shuan1,Pinyin.shuang1,Pinyin.shuang1,Pinyin.shuang1,Pinyin.shui2,Pinyin.shui2,Pinyin.shui2,Pinyin.shun1,Pinyin.shun1,Pinyin.shun1,Pinyin.shuo1,Pinyin.shuo1,Pinyin.shuo1,Pinyin.si1,Pinyin.si1,Pinyin.si1,Pinyin.song1,Pinyin.song1,Pinyin.song1,Pinyin.sou1,Pinyin.sou1,Pinyin.sou1,Pinyin.su1,Pinyin.su1,Pinyin.su1,Pinyin.suan1,Pinyin.suan1,Pinyin.suan1,Pinyin.sui1,Pinyin.sui1,Pinyin.sui1,Pinyin.sui1,Pinyin.sun1,Pinyin.sun1,Pinyin.sun1,Pinyin.suo1,Pinyin.suo1,Pinyin.suo1,Pinyin.ta1,Pinyin.ta1,Pinyin.ta1,Pinyin.ta1,Pinyin.tai1,Pinyin.tai1,Pinyin.tai1,Pinyin.tai1,Pinyin.tan1,Pinyin.tan1,Pinyin.tan1,Pinyin.tan1,Pinyin.tang1,Pinyin.tang1,Pinyin.tang1,Pinyin.tang1,Pinyin.tao1,Pinyin.tao1,Pinyin.tao1,Pinyin.tao1,Pinyin.te4,Pinyin.teng1,Pinyin.teng1,Pinyin.teng1,Pinyin.ti1,Pinyin.ti1,Pinyin.ti1,Pinyin.ti1,Pinyin.tian1,Pinyin.tian1,Pinyin.tian1,Pinyin.tian1,Pinyin.tian1,Pinyin.tiao1,Pinyin.tiao1,Pinyin.tiao1,Pinyin.tiao1,Pinyin.tie1,Pinyin.tie1,Pinyin.tie1,Pinyin.tie1,Pinyin.ting1,Pinyin.ting1,Pinyin.ting1,Pinyin.ting1,Pinyin.tong1,Pinyin.tong1,Pinyin.tong1,Pinyin.tong1,Pinyin.tou1,Pinyin.tou1,Pinyin.tou1,Pinyin.tou1,Pinyin.tou1,Pinyin.tu1,Pinyin.tu1,Pinyin.tu1,Pinyin.tu1,Pinyin.tu1,Pinyin.tuan1,Pinyin.tuan1,Pinyin.tuan1,Pinyin.tuan1,Pinyin.tui1,Pinyin.tui1,Pinyin.tui1,Pinyin.tui1,Pinyin.tun1,Pinyin.tun1,Pinyin.tun1,Pinyin.tun1,Pinyin.tun1,Pinyin.tuo1,Pinyin.tuo1,Pinyin.tuo1,Pinyin.tuo1,Pinyin.wa1,Pinyin.wa1,Pinyin.wa1,Pinyin.wa1,Pinyin.wa1,Pinyin.wai1,Pinyin.wai1,Pinyin.wai1,Pinyin.wan1,Pinyin.wan1,Pinyin.wan1,Pinyin.wan1,Pinyin.wang1,Pinyin.wang1,Pinyin.wang1,Pinyin.wang1,Pinyin.wei1,Pinyin.wei1,Pinyin.wei1,Pinyin.wei1,Pinyin.wen1,Pinyin.wen1,Pinyin.wen1,Pinyin.wen1,Pinyin.weng1,Pinyin.weng1,Pinyin.weng1,Pinyin.wo1,Pinyin.wo1,Pinyin.wo1,Pinyin.wu1,Pinyin.wu1,Pinyin.wu1,Pinyin.wu1,Pinyin.xi1,Pinyin.xi1,Pinyin.xi1,Pinyin.xi1,Pinyin.xia1,Pinyin.xia1,Pinyin.xia1,Pinyin.xia1,Pinyin.xian1,Pinyin.xian1,Pinyin.xian1,Pinyin.xian1,Pinyin.xiang1,Pinyin.xiang1,Pinyin.xiang1,Pinyin.xiang1,Pinyin.xiao1,Pinyin.xiao1,Pinyin.xiao1,Pinyin.xiao1,Pinyin.xie1,Pinyin.xie1,Pinyin.xie1,Pinyin.xie1,Pinyin.xin1,Pinyin.xin1,Pinyin.xin1,Pinyin.xin1,Pinyin.xing1,Pinyin.xing1,Pinyin.xing1,Pinyin.xing1,Pinyin.xiong1,Pinyin.xiong1,Pinyin.xiong1,Pinyin.xiong1,Pinyin.xiu1,Pinyin.xiu1,Pinyin.xiu1,Pinyin.xiu1,Pinyin.xu1,Pinyin.xu1,Pinyin.xu1,Pinyin.xu1,Pinyin.xu1,Pinyin.xuan1,Pinyin.xuan1,Pinyin.xuan1,Pinyin.xuan1,Pinyin.xue1,Pinyin.xue1,Pinyin.xue1,Pinyin.xue1,Pinyin.xun1,Pinyin.xun1,Pinyin.xun1,Pinyin.ya1,Pinyin.ya1,Pinyin.ya1,Pinyin.ya1,Pinyin.ya1,Pinyin.yai2,Pinyin.yan1,Pinyin.yan1,Pinyin.yan1,Pinyin.yan1,Pinyin.yang1,Pinyin.yang1,Pinyin.yang1,Pinyin.yang1,Pinyin.yao1,Pinyin.yao1,Pinyin.yao1,Pinyin.yao1,Pinyin.ye1,Pinyin.ye1,Pinyin.ye1,Pinyin.ye1,Pinyin.ye1,Pinyin.yi1,Pinyin.yi1,Pinyin.yi1,Pinyin.yi1,Pinyin.yi1,Pinyin.yin1,Pinyin.yin1,Pinyin.yin1,Pinyin.yin1,Pinyin.ying1,Pinyin.ying1,Pinyin.ying1,Pinyin.ying1,Pinyin.yo1,Pinyin.yo1,Pinyin.yong1,Pinyin.yong1,Pinyin.yong1,Pinyin.yong1,Pinyin.you1,Pinyin.you1,Pinyin.you1,Pinyin.you1,Pinyin.yu1,Pinyin.yu1,Pinyin.yu1,Pinyin.yu1,Pinyin.yuan1,Pinyin.yuan1,Pinyin.yuan1,Pinyin.yuan1,Pinyin.yue1,Pinyin.yue1,Pinyin.yue1,Pinyin.yun1,Pinyin.yun1,Pinyin.yun1,Pinyin.yun1,Pinyin.za1,Pinyin.za1,Pinyin.za1,Pinyin.zai1,Pinyin.zai1,Pinyin.zai1,Pinyin.zan1,Pinyin.zan1,Pinyin.zan1,Pinyin.zan1,Pinyin.zang1,Pinyin.zang1,Pinyin.zang1,Pinyin.zang1,Pinyin.zao1,Pinyin.zao1,Pinyin.zao1,Pinyin.zao1,Pinyin.ze2,Pinyin.ze2,Pinyin.zei2,Pinyin.zen1,Pinyin.zen1,Pinyin.zen1,Pinyin.zeng1,Pinyin.zeng1,Pinyin.zha1,Pinyin.zha1,Pinyin.zha1,Pinyin.zha1,Pinyin.zhai1,Pinyin.zhai1,Pinyin.zhai1,Pinyin.zhai1,Pinyin.zhan1,Pinyin.zhan1,Pinyin.zhan1,Pinyin.zhan1,Pinyin.zhang1,Pinyin.zhang1,Pinyin.zhang1,Pinyin.zhao1,Pinyin.zhao1,Pinyin.zhao1,Pinyin.zhao1,Pinyin.zhe1,Pinyin.zhe1,Pinyin.zhe1,Pinyin.zhe1,Pinyin.zhe1,Pinyin.zhei4,Pinyin.zhen1,Pinyin.zhen1,Pinyin.zhen1,Pinyin.zheng1,Pinyin.zheng1,Pinyin.zheng1,Pinyin.zhi1,Pinyin.zhi1,Pinyin.zhi1,Pinyin.zhi1,Pinyin.zhong1,Pinyin.zhong1,Pinyin.zhong1,Pinyin.zhou1,Pinyin.zhou1,Pinyin.zhou1,Pinyin.zhou1,Pinyin.zhu1,Pinyin.zhu1,Pinyin.zhu1,Pinyin.zhu1,Pinyin.zhua1,Pinyin.zhua1,Pinyin.zhuai1,Pinyin.zhuai1,Pinyin.zhuai1,Pinyin.zhuan1,Pinyin.zhuan1,Pinyin.zhuan1,Pinyin.zhuang1,Pinyin.zhuang1,Pinyin.zhuang1,Pinyin.zhui1,Pinyin.zhui1,Pinyin.zhui1,Pinyin.zhun1,Pinyin.zhun1,Pinyin.zhun1,Pinyin.zhuo1,Pinyin.zhuo1,Pinyin.zhuo1,Pinyin.zhuo1,Pinyin.zi1,Pinyin.zi1,Pinyin.zi1,Pinyin.zi1,Pinyin.zi1,Pinyin.zong1,Pinyin.zong1,Pinyin.zong1,Pinyin.zou1,Pinyin.zou1,Pinyin.zou1,Pinyin.zu1,Pinyin.zu1,Pinyin.zu1,Pinyin.zu1,Pinyin.zuan1,Pinyin.zuan1,Pinyin.zuan1,Pinyin.zui1,Pinyin.zui1,Pinyin.zui1,Pinyin.zun1,Pinyin.zun1,Pinyin.zun1,Pinyin.zuo1,Pinyin.zuo1,Pinyin.zuo1,Pinyin.zuo1,Pinyin.zuo1,Pinyin.none5,};
    static
    {
        trie = new Trie().remainLongest();
        map = new TreeMap<>();
        int end = Pinyin.none5.ordinal();
        for (int i = 0; i < end; ++i)
        {
            Pinyin pinyin = PinyinDictionary.pinyins[i];
            String pinyinWithoutTone = pinyin.getPinyinWithoutTone();
            String head = pinyin.getHeadString();
            trie.addKeyword(pinyinWithoutTone);
            trie.addKeyword(head);
            map.put(pinyinWithoutTone, pinyin);
            map.put(head, pinyin);
            map.put(pinyin.toString(), pinyin);
        }
    }

    /**
     * 将拼音文本转化为完整的拼音，支持汉英混合的杂乱文本，注意如果混用拼音和输入法头的话，并不会有多高的准确率，声调也不会准的
     * @param complexText
     * @return
     */
    public static Pinyin[] convert2Array(String complexText, boolean removeTone)
    {
        return PinyinUtil.convertList2Array(convert(complexText, removeTone));
    }

    /**
     * 文本转拼音
     * @param complexText
     * @return
     */
    public static List<Pinyin> convert(String complexText)
    {
        List<Pinyin> pinyinList = new LinkedList<>();
        Collection<Token> tokenize = trie.tokenize(complexText);
//        System.out.println(tokenize);
        for (Token token : tokenize)
        {
            String fragment = token.getFragment();
            if (token.isMatch())
            {
                // 是拼音或拼音的一部分，用map转
                pinyinList.add(convertSingle(fragment));
            }
            else
            {
                pinyinList.addAll(PinyinDictionary.convertToPinyin(fragment));
            }
        }

        return pinyinList;
    }

    /**
     * 文本转拼音
     * @param complexText 文本
     * @param removeTone 是否将所有的音调都同一化
     * @return
     */
    public static List<Pinyin> convert(String complexText, boolean removeTone)
    {
        List<Pinyin> pinyinList = convert(complexText);
        if (removeTone)
        {
            makeToneToTheSame(pinyinList);
        }
        return pinyinList;
    }

    /**
     * 将混合文本转为拼音
     * @param complexText 混合汉字、拼音、输入法头的文本，比如“飞流zh下sqianch”
     * @param removeTone
     * @return 一个键值对，键为拼音列表，值为类型（true表示这是一个拼音，false表示这是一个输入法头）
     */
    public static Pair<List<Pinyin>, List<Boolean>> convert2Pair(String complexText, boolean removeTone)
    {
        List<Pinyin> pinyinList = new LinkedList<>();
        List<Boolean> booleanList = new LinkedList<>();
        Collection<Token> tokenize = trie.tokenize(complexText);
        for (Token token : tokenize)
        {
            String fragment = token.getFragment();
            if (token.isMatch())
            {
                // 是拼音或拼音的一部分，用map转
                Pinyin pinyin = convertSingle(fragment);
                pinyinList.add(pinyin);
                if (fragment.length() == pinyin.getPinyinWithoutTone().length())
                {
                    booleanList.add(true);
                }
                else
                {
                    booleanList.add(false);
                }
            }
            else
            {
                List<Pinyin> pinyinListFragment = PinyinDictionary.convertToPinyin(fragment);
                pinyinList.addAll(pinyinListFragment);
                for (int i = 0; i < pinyinListFragment.size(); ++i)
                {
                    booleanList.add(true);
                }
            }
        }
        makeToneToTheSame(pinyinList);
        return new Pair<>(pinyinList, booleanList);
    }

    /**
     * 将单个音节转为拼音
     * @param single
     * @return
     */
    public static Pinyin convertSingle(String single)
    {
        Pinyin pinyin = map.get(single);
        if (pinyin == null) return Pinyin.none5;

        return pinyin;
    }

    /**
     * 将拼音的音调统统转为1调或者最小的音调
     * @param p
     * @return
     */
    public static Pinyin convert(Pinyin p)
    {
        return tone2tone1[p.ordinal()];
    }

    /**
     * 将所有音调都转为1
     * @param pinyinList
     * @return
     */
    public static List<Pinyin> makeToneToTheSame(List<Pinyin> pinyinList)
    {
        ListIterator<Pinyin> listIterator = pinyinList.listIterator();
        while (listIterator.hasNext())
        {
            listIterator.set(convert(listIterator.next()));
        }

        return pinyinList;
    }
}
