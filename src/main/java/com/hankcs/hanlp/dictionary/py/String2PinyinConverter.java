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

import com.hankcs.hanlp.algorithm.ahocorasick.trie.Token;
import com.hankcs.hanlp.algorithm.ahocorasick.trie.Trie;
import com.hankcs.hanlp.collection.dartsclone.Pair;

import java.util.*;

/**
 * 汉字转拼音，提供拼音字符串转拼音，支持汉英混合的杂乱文本
 * @author hankcs
 */
public class String2PinyinConverter
{
    static Trie trie;
    /**
     * 将拼音和输入法头转为Pyinyin的map
     */
    static Map<String, Pinyin> map;
    /**
     * 将音调统一换为轻声，下标为拼音的ordinal，值为音调5或最大值
     */
    public static Pinyin[] tone2tone5 = new Pinyin[]{Pinyin.a5,Pinyin.a5,Pinyin.a5,Pinyin.a5,Pinyin.a5,Pinyin.ai4,Pinyin.ai4,Pinyin.ai4,Pinyin.ai4,Pinyin.an4,Pinyin.an4,Pinyin.an4,Pinyin.an4,Pinyin.ang4,Pinyin.ang4,Pinyin.ang4,Pinyin.ang4,Pinyin.ao4,Pinyin.ao4,Pinyin.ao4,Pinyin.ao4,Pinyin.ba5,Pinyin.ba5,Pinyin.ba5,Pinyin.ba5,Pinyin.ba5,Pinyin.bai4,Pinyin.bai4,Pinyin.bai4,Pinyin.bai4,Pinyin.ban4,Pinyin.ban4,Pinyin.ban4,Pinyin.bang4,Pinyin.bang4,Pinyin.bang4,Pinyin.bao4,Pinyin.bao4,Pinyin.bao4,Pinyin.bao4,Pinyin.bei5,Pinyin.bei5,Pinyin.bei5,Pinyin.bei5,Pinyin.ben4,Pinyin.ben4,Pinyin.ben4,Pinyin.beng4,Pinyin.beng4,Pinyin.beng4,Pinyin.beng4,Pinyin.bi4,Pinyin.bi4,Pinyin.bi4,Pinyin.bi4,Pinyin.bian5,Pinyin.bian5,Pinyin.bian5,Pinyin.bian5,Pinyin.biao4,Pinyin.biao4,Pinyin.biao4,Pinyin.biao4,Pinyin.bie4,Pinyin.bie4,Pinyin.bie4,Pinyin.bie4,Pinyin.bin4,Pinyin.bin4,Pinyin.bin4,Pinyin.bing4,Pinyin.bing4,Pinyin.bing4,Pinyin.bo5,Pinyin.bo5,Pinyin.bo5,Pinyin.bo5,Pinyin.bo5,Pinyin.bu4,Pinyin.bu4,Pinyin.bu4,Pinyin.bu4,Pinyin.ca4,Pinyin.ca4,Pinyin.ca4,Pinyin.cai4,Pinyin.cai4,Pinyin.cai4,Pinyin.cai4,Pinyin.can4,Pinyin.can4,Pinyin.can4,Pinyin.can4,Pinyin.cang4,Pinyin.cang4,Pinyin.cang4,Pinyin.cang4,Pinyin.cao4,Pinyin.cao4,Pinyin.cao4,Pinyin.cao4,Pinyin.ce4,Pinyin.cen2,Pinyin.cen2,Pinyin.ceng4,Pinyin.ceng4,Pinyin.ceng4,Pinyin.cha5,Pinyin.cha5,Pinyin.cha5,Pinyin.cha5,Pinyin.cha5,Pinyin.chai4,Pinyin.chai4,Pinyin.chai4,Pinyin.chai4,Pinyin.chan4,Pinyin.chan4,Pinyin.chan4,Pinyin.chan4,Pinyin.chang5,Pinyin.chang5,Pinyin.chang5,Pinyin.chang5,Pinyin.chang5,Pinyin.chao4,Pinyin.chao4,Pinyin.chao4,Pinyin.chao4,Pinyin.che4,Pinyin.che4,Pinyin.che4,Pinyin.chen5,Pinyin.chen5,Pinyin.chen5,Pinyin.chen5,Pinyin.chen5,Pinyin.cheng4,Pinyin.cheng4,Pinyin.cheng4,Pinyin.cheng4,Pinyin.chi5,Pinyin.chi5,Pinyin.chi5,Pinyin.chi5,Pinyin.chi5,Pinyin.chong4,Pinyin.chong4,Pinyin.chong4,Pinyin.chong4,Pinyin.chou5,Pinyin.chou5,Pinyin.chou5,Pinyin.chou5,Pinyin.chou5,Pinyin.chu5,Pinyin.chu5,Pinyin.chu5,Pinyin.chu5,Pinyin.chu5,Pinyin.chua1,Pinyin.chuai4,Pinyin.chuai4,Pinyin.chuai4,Pinyin.chuai4,Pinyin.chuan4,Pinyin.chuan4,Pinyin.chuan4,Pinyin.chuan4,Pinyin.chuang4,Pinyin.chuang4,Pinyin.chuang4,Pinyin.chuang4,Pinyin.chui4,Pinyin.chui4,Pinyin.chui4,Pinyin.chun3,Pinyin.chun3,Pinyin.chun3,Pinyin.chuo5,Pinyin.chuo5,Pinyin.chuo5,Pinyin.chuo5,Pinyin.ci4,Pinyin.ci4,Pinyin.ci4,Pinyin.ci4,Pinyin.cong4,Pinyin.cong4,Pinyin.cong4,Pinyin.cou4,Pinyin.cou4,Pinyin.cu4,Pinyin.cu4,Pinyin.cu4,Pinyin.cu4,Pinyin.cuan4,Pinyin.cuan4,Pinyin.cuan4,Pinyin.cui4,Pinyin.cui4,Pinyin.cui4,Pinyin.cui4,Pinyin.cun4,Pinyin.cun4,Pinyin.cun4,Pinyin.cun4,Pinyin.cuo4,Pinyin.cuo4,Pinyin.cuo4,Pinyin.cuo4,Pinyin.da5,Pinyin.da5,Pinyin.da5,Pinyin.da5,Pinyin.da5,Pinyin.dai4,Pinyin.dai4,Pinyin.dai4,Pinyin.dan4,Pinyin.dan4,Pinyin.dan4,Pinyin.dan4,Pinyin.dang4,Pinyin.dang4,Pinyin.dang4,Pinyin.dao4,Pinyin.dao4,Pinyin.dao4,Pinyin.dao4,Pinyin.de5,Pinyin.de5,Pinyin.de5,Pinyin.dei3,Pinyin.dei3,Pinyin.den4,Pinyin.den4,Pinyin.deng4,Pinyin.deng4,Pinyin.deng4,Pinyin.di4,Pinyin.di4,Pinyin.di4,Pinyin.di4,Pinyin.dia3,Pinyin.dian4,Pinyin.dian4,Pinyin.dian4,Pinyin.diao4,Pinyin.diao4,Pinyin.diao4,Pinyin.die4,Pinyin.die4,Pinyin.die4,Pinyin.ding4,Pinyin.ding4,Pinyin.ding4,Pinyin.ding4,Pinyin.diu1,Pinyin.dong4,Pinyin.dong4,Pinyin.dong4,Pinyin.dou4,Pinyin.dou4,Pinyin.dou4,Pinyin.dou4,Pinyin.du4,Pinyin.du4,Pinyin.du4,Pinyin.du4,Pinyin.duan4,Pinyin.duan4,Pinyin.duan4,Pinyin.dui4,Pinyin.dui4,Pinyin.dui4,Pinyin.dun4,Pinyin.dun4,Pinyin.dun4,Pinyin.dun4,Pinyin.duo5,Pinyin.duo5,Pinyin.duo5,Pinyin.duo5,Pinyin.duo5,Pinyin.e4,Pinyin.e4,Pinyin.e4,Pinyin.e4,Pinyin.ei4,Pinyin.ei4,Pinyin.ei4,Pinyin.ei4,Pinyin.en4,Pinyin.en4,Pinyin.en4,Pinyin.eng1,Pinyin.er5,Pinyin.er5,Pinyin.er5,Pinyin.er5,Pinyin.fa4,Pinyin.fa4,Pinyin.fa4,Pinyin.fa4,Pinyin.fan4,Pinyin.fan4,Pinyin.fan4,Pinyin.fan4,Pinyin.fang5,Pinyin.fang5,Pinyin.fang5,Pinyin.fang5,Pinyin.fang5,Pinyin.fei4,Pinyin.fei4,Pinyin.fei4,Pinyin.fei4,Pinyin.fen4,Pinyin.fen4,Pinyin.fen4,Pinyin.fen4,Pinyin.feng4,Pinyin.feng4,Pinyin.feng4,Pinyin.feng4,Pinyin.fiao4,Pinyin.fo2,Pinyin.fou4,Pinyin.fou4,Pinyin.fou4,Pinyin.fou4,Pinyin.fu5,Pinyin.fu5,Pinyin.fu5,Pinyin.fu5,Pinyin.fu5,Pinyin.ga4,Pinyin.ga4,Pinyin.ga4,Pinyin.ga4,Pinyin.gai4,Pinyin.gai4,Pinyin.gai4,Pinyin.gan4,Pinyin.gan4,Pinyin.gan4,Pinyin.gan4,Pinyin.gang4,Pinyin.gang4,Pinyin.gang4,Pinyin.gao4,Pinyin.gao4,Pinyin.gao4,Pinyin.ge4,Pinyin.ge4,Pinyin.ge4,Pinyin.ge4,Pinyin.gei3,Pinyin.gen4,Pinyin.gen4,Pinyin.gen4,Pinyin.gen4,Pinyin.geng4,Pinyin.geng4,Pinyin.geng4,Pinyin.gong4,Pinyin.gong4,Pinyin.gong4,Pinyin.gou4,Pinyin.gou4,Pinyin.gou4,Pinyin.gu4,Pinyin.gu4,Pinyin.gu4,Pinyin.gu4,Pinyin.gua4,Pinyin.gua4,Pinyin.gua4,Pinyin.guai4,Pinyin.guai4,Pinyin.guai4,Pinyin.guai4,Pinyin.guan4,Pinyin.guan4,Pinyin.guan4,Pinyin.guang4,Pinyin.guang4,Pinyin.guang4,Pinyin.gui4,Pinyin.gui4,Pinyin.gui4,Pinyin.gui4,Pinyin.gun4,Pinyin.gun4,Pinyin.gun4,Pinyin.guo5,Pinyin.guo5,Pinyin.guo5,Pinyin.guo5,Pinyin.guo5,Pinyin.ha4,Pinyin.ha4,Pinyin.ha4,Pinyin.ha4,Pinyin.hai4,Pinyin.hai4,Pinyin.hai4,Pinyin.hai4,Pinyin.han5,Pinyin.han5,Pinyin.han5,Pinyin.han5,Pinyin.han5,Pinyin.hang4,Pinyin.hang4,Pinyin.hang4,Pinyin.hang4,Pinyin.hao4,Pinyin.hao4,Pinyin.hao4,Pinyin.hao4,Pinyin.he4,Pinyin.he4,Pinyin.he4,Pinyin.hei1,Pinyin.hen4,Pinyin.hen4,Pinyin.hen4,Pinyin.hen4,Pinyin.heng4,Pinyin.heng4,Pinyin.heng4,Pinyin.hong4,Pinyin.hong4,Pinyin.hong4,Pinyin.hong4,Pinyin.hou4,Pinyin.hou4,Pinyin.hou4,Pinyin.hou4,Pinyin.hu4,Pinyin.hu4,Pinyin.hu4,Pinyin.hu4,Pinyin.hua4,Pinyin.hua4,Pinyin.hua4,Pinyin.hua4,Pinyin.huai4,Pinyin.huai4,Pinyin.huai4,Pinyin.huan4,Pinyin.huan4,Pinyin.huan4,Pinyin.huan4,Pinyin.huang4,Pinyin.huang4,Pinyin.huang4,Pinyin.huang4,Pinyin.hui4,Pinyin.hui4,Pinyin.hui4,Pinyin.hui4,Pinyin.hun4,Pinyin.hun4,Pinyin.hun4,Pinyin.hun4,Pinyin.huo5,Pinyin.huo5,Pinyin.huo5,Pinyin.huo5,Pinyin.huo5,Pinyin.ja4,Pinyin.ji5,Pinyin.ji5,Pinyin.ji5,Pinyin.ji5,Pinyin.ji5,Pinyin.jia5,Pinyin.jia5,Pinyin.jia5,Pinyin.jia5,Pinyin.jia5,Pinyin.jian4,Pinyin.jian4,Pinyin.jian4,Pinyin.jiang4,Pinyin.jiang4,Pinyin.jiang4,Pinyin.jiao4,Pinyin.jiao4,Pinyin.jiao4,Pinyin.jiao4,Pinyin.jie5,Pinyin.jie5,Pinyin.jie5,Pinyin.jie5,Pinyin.jie5,Pinyin.jin4,Pinyin.jin4,Pinyin.jin4,Pinyin.jing4,Pinyin.jing4,Pinyin.jing4,Pinyin.jiong4,Pinyin.jiong4,Pinyin.jiong4,Pinyin.jiu4,Pinyin.jiu4,Pinyin.jiu4,Pinyin.ju5,Pinyin.ju5,Pinyin.ju5,Pinyin.ju5,Pinyin.ju5,Pinyin.juan4,Pinyin.juan4,Pinyin.juan4,Pinyin.juan4,Pinyin.jue4,Pinyin.jue4,Pinyin.jue4,Pinyin.jue4,Pinyin.jun4,Pinyin.jun4,Pinyin.jun4,Pinyin.ka4,Pinyin.ka4,Pinyin.ka4,Pinyin.kai4,Pinyin.kai4,Pinyin.kai4,Pinyin.kan4,Pinyin.kan4,Pinyin.kan4,Pinyin.kang4,Pinyin.kang4,Pinyin.kang4,Pinyin.kang4,Pinyin.kao4,Pinyin.kao4,Pinyin.kao4,Pinyin.kao4,Pinyin.ke5,Pinyin.ke5,Pinyin.ke5,Pinyin.ke5,Pinyin.ke5,Pinyin.kei1,Pinyin.ken4,Pinyin.ken4,Pinyin.keng3,Pinyin.keng3,Pinyin.kong4,Pinyin.kong4,Pinyin.kong4,Pinyin.kou4,Pinyin.kou4,Pinyin.kou4,Pinyin.ku4,Pinyin.ku4,Pinyin.ku4,Pinyin.kua4,Pinyin.kua4,Pinyin.kua4,Pinyin.kuai4,Pinyin.kuai4,Pinyin.kuai4,Pinyin.kuan3,Pinyin.kuan3,Pinyin.kuang4,Pinyin.kuang4,Pinyin.kuang4,Pinyin.kuang4,Pinyin.kui4,Pinyin.kui4,Pinyin.kui4,Pinyin.kui4,Pinyin.kun4,Pinyin.kun4,Pinyin.kun4,Pinyin.kuo4,Pinyin.kuo4,Pinyin.la5,Pinyin.la5,Pinyin.la5,Pinyin.la5,Pinyin.la5,Pinyin.lai4,Pinyin.lai4,Pinyin.lai4,Pinyin.lan5,Pinyin.lan5,Pinyin.lan5,Pinyin.lan5,Pinyin.lang4,Pinyin.lang4,Pinyin.lang4,Pinyin.lang4,Pinyin.lao4,Pinyin.lao4,Pinyin.lao4,Pinyin.lao4,Pinyin.le5,Pinyin.le5,Pinyin.le5,Pinyin.lei5,Pinyin.lei5,Pinyin.lei5,Pinyin.lei5,Pinyin.lei5,Pinyin.leng4,Pinyin.leng4,Pinyin.leng4,Pinyin.leng4,Pinyin.li5,Pinyin.li5,Pinyin.li5,Pinyin.li5,Pinyin.li5,Pinyin.lia3,Pinyin.lian4,Pinyin.lian4,Pinyin.lian4,Pinyin.lian4,Pinyin.liang5,Pinyin.liang5,Pinyin.liang5,Pinyin.liang5,Pinyin.liao4,Pinyin.liao4,Pinyin.liao4,Pinyin.liao4,Pinyin.lie5,Pinyin.lie5,Pinyin.lie5,Pinyin.lie5,Pinyin.lie5,Pinyin.lin4,Pinyin.lin4,Pinyin.lin4,Pinyin.lin4,Pinyin.ling4,Pinyin.ling4,Pinyin.ling4,Pinyin.ling4,Pinyin.liu4,Pinyin.liu4,Pinyin.liu4,Pinyin.liu4,Pinyin.lo5,Pinyin.long4,Pinyin.long4,Pinyin.long4,Pinyin.long4,Pinyin.lou5,Pinyin.lou5,Pinyin.lou5,Pinyin.lou5,Pinyin.lou5,Pinyin.lu5,Pinyin.lu5,Pinyin.lu5,Pinyin.lu5,Pinyin.lu5,Pinyin.luan4,Pinyin.luan4,Pinyin.luan4,Pinyin.lun4,Pinyin.lun4,Pinyin.lun4,Pinyin.lun4,Pinyin.luo5,Pinyin.luo5,Pinyin.luo5,Pinyin.luo5,Pinyin.luo5,Pinyin.lv4,Pinyin.lv4,Pinyin.lv4,Pinyin.lve4,Pinyin.lve4,Pinyin.ma5,Pinyin.ma5,Pinyin.ma5,Pinyin.ma5,Pinyin.ma5,Pinyin.mai4,Pinyin.mai4,Pinyin.mai4,Pinyin.man4,Pinyin.man4,Pinyin.man4,Pinyin.man4,Pinyin.mang3,Pinyin.mang3,Pinyin.mang3,Pinyin.mao4,Pinyin.mao4,Pinyin.mao4,Pinyin.mao4,Pinyin.me5,Pinyin.me5,Pinyin.me5,Pinyin.mei4,Pinyin.mei4,Pinyin.mei4,Pinyin.men5,Pinyin.men5,Pinyin.men5,Pinyin.men5,Pinyin.men5,Pinyin.meng4,Pinyin.meng4,Pinyin.meng4,Pinyin.meng4,Pinyin.mi4,Pinyin.mi4,Pinyin.mi4,Pinyin.mi4,Pinyin.mian4,Pinyin.mian4,Pinyin.mian4,Pinyin.miao4,Pinyin.miao4,Pinyin.miao4,Pinyin.miao4,Pinyin.mie4,Pinyin.mie4,Pinyin.min3,Pinyin.min3,Pinyin.ming4,Pinyin.ming4,Pinyin.ming4,Pinyin.miu4,Pinyin.mo5,Pinyin.mo5,Pinyin.mo5,Pinyin.mo5,Pinyin.mo5,Pinyin.mou4,Pinyin.mou4,Pinyin.mou4,Pinyin.mou4,Pinyin.mu4,Pinyin.mu4,Pinyin.mu4,Pinyin.na5,Pinyin.na5,Pinyin.na5,Pinyin.na5,Pinyin.na5,Pinyin.nai4,Pinyin.nai4,Pinyin.nai4,Pinyin.nan4,Pinyin.nan4,Pinyin.nan4,Pinyin.nan4,Pinyin.nang4,Pinyin.nang4,Pinyin.nang4,Pinyin.nang4,Pinyin.nao4,Pinyin.nao4,Pinyin.nao4,Pinyin.nao4,Pinyin.ne5,Pinyin.ne5,Pinyin.ne5,Pinyin.nei4,Pinyin.nei4,Pinyin.nen4,Pinyin.nen4,Pinyin.nen4,Pinyin.neng3,Pinyin.neng3,Pinyin.ni4,Pinyin.ni4,Pinyin.ni4,Pinyin.ni4,Pinyin.nian4,Pinyin.nian4,Pinyin.nian4,Pinyin.nian4,Pinyin.niang4,Pinyin.niang4,Pinyin.niao4,Pinyin.niao4,Pinyin.nie4,Pinyin.nie4,Pinyin.nie4,Pinyin.nie4,Pinyin.nin3,Pinyin.nin3,Pinyin.ning4,Pinyin.ning4,Pinyin.ning4,Pinyin.niu4,Pinyin.niu4,Pinyin.niu4,Pinyin.niu4,Pinyin.nong4,Pinyin.nong4,Pinyin.nong4,Pinyin.nou4,Pinyin.nou4,Pinyin.nu4,Pinyin.nu4,Pinyin.nu4,Pinyin.nuan4,Pinyin.nuan4,Pinyin.nuan4,Pinyin.nun4,Pinyin.nun4,Pinyin.nuo4,Pinyin.nuo4,Pinyin.nuo4,Pinyin.nv4,Pinyin.nv4,Pinyin.nve4,Pinyin.o5,Pinyin.o5,Pinyin.o5,Pinyin.o5,Pinyin.o5,Pinyin.ou5,Pinyin.ou5,Pinyin.ou5,Pinyin.ou5,Pinyin.ou5,Pinyin.pa4,Pinyin.pa4,Pinyin.pa4,Pinyin.pai4,Pinyin.pai4,Pinyin.pai4,Pinyin.pai4,Pinyin.pan4,Pinyin.pan4,Pinyin.pan4,Pinyin.pan4,Pinyin.pang5,Pinyin.pang5,Pinyin.pang5,Pinyin.pang5,Pinyin.pang5,Pinyin.pao4,Pinyin.pao4,Pinyin.pao4,Pinyin.pao4,Pinyin.pei4,Pinyin.pei4,Pinyin.pei4,Pinyin.pei4,Pinyin.pen5,Pinyin.pen5,Pinyin.pen5,Pinyin.pen5,Pinyin.pen5,Pinyin.peng4,Pinyin.peng4,Pinyin.peng4,Pinyin.peng4,Pinyin.pi5,Pinyin.pi5,Pinyin.pi5,Pinyin.pi5,Pinyin.pi5,Pinyin.pian4,Pinyin.pian4,Pinyin.pian4,Pinyin.pian4,Pinyin.piao4,Pinyin.piao4,Pinyin.piao4,Pinyin.piao4,Pinyin.pie4,Pinyin.pie4,Pinyin.pie4,Pinyin.pin4,Pinyin.pin4,Pinyin.pin4,Pinyin.pin4,Pinyin.ping4,Pinyin.ping4,Pinyin.ping4,Pinyin.ping4,Pinyin.po5,Pinyin.po5,Pinyin.po5,Pinyin.po5,Pinyin.po5,Pinyin.pou4,Pinyin.pou4,Pinyin.pou4,Pinyin.pou4,Pinyin.pu4,Pinyin.pu4,Pinyin.pu4,Pinyin.pu4,Pinyin.qi5,Pinyin.qi5,Pinyin.qi5,Pinyin.qi5,Pinyin.qi5,Pinyin.qia4,Pinyin.qia4,Pinyin.qia4,Pinyin.qian5,Pinyin.qian5,Pinyin.qian5,Pinyin.qian5,Pinyin.qian5,Pinyin.qiang4,Pinyin.qiang4,Pinyin.qiang4,Pinyin.qiang4,Pinyin.qiao4,Pinyin.qiao4,Pinyin.qiao4,Pinyin.qiao4,Pinyin.qie4,Pinyin.qie4,Pinyin.qie4,Pinyin.qie4,Pinyin.qin4,Pinyin.qin4,Pinyin.qin4,Pinyin.qin4,Pinyin.qing4,Pinyin.qing4,Pinyin.qing4,Pinyin.qing4,Pinyin.qiong3,Pinyin.qiong3,Pinyin.qiong3,Pinyin.qiu4,Pinyin.qiu4,Pinyin.qiu4,Pinyin.qiu4,Pinyin.qu4,Pinyin.qu4,Pinyin.qu4,Pinyin.qu4,Pinyin.quan4,Pinyin.quan4,Pinyin.quan4,Pinyin.quan4,Pinyin.que4,Pinyin.que4,Pinyin.que4,Pinyin.qun3,Pinyin.qun3,Pinyin.qun3,Pinyin.ran3,Pinyin.ran3,Pinyin.rang4,Pinyin.rang4,Pinyin.rang4,Pinyin.rang4,Pinyin.rao4,Pinyin.rao4,Pinyin.rao4,Pinyin.re4,Pinyin.re4,Pinyin.re4,Pinyin.ren4,Pinyin.ren4,Pinyin.ren4,Pinyin.reng4,Pinyin.reng4,Pinyin.reng4,Pinyin.ri4,Pinyin.rong4,Pinyin.rong4,Pinyin.rong4,Pinyin.rou4,Pinyin.rou4,Pinyin.rou4,Pinyin.ru4,Pinyin.ru4,Pinyin.ru4,Pinyin.ru4,Pinyin.ruan4,Pinyin.ruan4,Pinyin.ruan4,Pinyin.rui4,Pinyin.rui4,Pinyin.rui4,Pinyin.run4,Pinyin.run4,Pinyin.ruo4,Pinyin.ruo4,Pinyin.sa4,Pinyin.sa4,Pinyin.sa4,Pinyin.sai5,Pinyin.sai5,Pinyin.sai5,Pinyin.sai5,Pinyin.san5,Pinyin.san5,Pinyin.san5,Pinyin.san5,Pinyin.sang5,Pinyin.sang5,Pinyin.sang5,Pinyin.sang5,Pinyin.sao4,Pinyin.sao4,Pinyin.sao4,Pinyin.se4,Pinyin.se4,Pinyin.sen3,Pinyin.sen3,Pinyin.seng1,Pinyin.sha4,Pinyin.sha4,Pinyin.sha4,Pinyin.sha4,Pinyin.shai4,Pinyin.shai4,Pinyin.shai4,Pinyin.shan4,Pinyin.shan4,Pinyin.shan4,Pinyin.shan4,Pinyin.shang5,Pinyin.shang5,Pinyin.shang5,Pinyin.shang5,Pinyin.shang5,Pinyin.shao4,Pinyin.shao4,Pinyin.shao4,Pinyin.shao4,Pinyin.she4,Pinyin.she4,Pinyin.she4,Pinyin.she4,Pinyin.shei2,Pinyin.shen4,Pinyin.shen4,Pinyin.shen4,Pinyin.shen4,Pinyin.sheng4,Pinyin.sheng4,Pinyin.sheng4,Pinyin.sheng4,Pinyin.shi5,Pinyin.shi5,Pinyin.shi5,Pinyin.shi5,Pinyin.shi5,Pinyin.shou4,Pinyin.shou4,Pinyin.shou4,Pinyin.shou4,Pinyin.shu4,Pinyin.shu4,Pinyin.shu4,Pinyin.shu4,Pinyin.shua4,Pinyin.shua4,Pinyin.shua4,Pinyin.shuai4,Pinyin.shuai4,Pinyin.shuai4,Pinyin.shuan4,Pinyin.shuan4,Pinyin.shuang4,Pinyin.shuang4,Pinyin.shuang4,Pinyin.shui4,Pinyin.shui4,Pinyin.shui4,Pinyin.shun4,Pinyin.shun4,Pinyin.shun4,Pinyin.shuo4,Pinyin.shuo4,Pinyin.shuo4,Pinyin.si4,Pinyin.si4,Pinyin.si4,Pinyin.song4,Pinyin.song4,Pinyin.song4,Pinyin.sou4,Pinyin.sou4,Pinyin.sou4,Pinyin.su4,Pinyin.su4,Pinyin.su4,Pinyin.suan4,Pinyin.suan4,Pinyin.suan4,Pinyin.sui4,Pinyin.sui4,Pinyin.sui4,Pinyin.sui4,Pinyin.sun4,Pinyin.sun4,Pinyin.sun4,Pinyin.suo4,Pinyin.suo4,Pinyin.suo4,Pinyin.ta5,Pinyin.ta5,Pinyin.ta5,Pinyin.ta5,Pinyin.tai4,Pinyin.tai4,Pinyin.tai4,Pinyin.tai4,Pinyin.tan4,Pinyin.tan4,Pinyin.tan4,Pinyin.tan4,Pinyin.tang4,Pinyin.tang4,Pinyin.tang4,Pinyin.tang4,Pinyin.tao4,Pinyin.tao4,Pinyin.tao4,Pinyin.tao4,Pinyin.te4,Pinyin.teng4,Pinyin.teng4,Pinyin.teng4,Pinyin.ti4,Pinyin.ti4,Pinyin.ti4,Pinyin.ti4,Pinyin.tian5,Pinyin.tian5,Pinyin.tian5,Pinyin.tian5,Pinyin.tian5,Pinyin.tiao4,Pinyin.tiao4,Pinyin.tiao4,Pinyin.tiao4,Pinyin.tie4,Pinyin.tie4,Pinyin.tie4,Pinyin.tie4,Pinyin.ting4,Pinyin.ting4,Pinyin.ting4,Pinyin.ting4,Pinyin.tong4,Pinyin.tong4,Pinyin.tong4,Pinyin.tong4,Pinyin.tou5,Pinyin.tou5,Pinyin.tou5,Pinyin.tou5,Pinyin.tou5,Pinyin.tu5,Pinyin.tu5,Pinyin.tu5,Pinyin.tu5,Pinyin.tu5,Pinyin.tuan4,Pinyin.tuan4,Pinyin.tuan4,Pinyin.tuan4,Pinyin.tui4,Pinyin.tui4,Pinyin.tui4,Pinyin.tui4,Pinyin.tun5,Pinyin.tun5,Pinyin.tun5,Pinyin.tun5,Pinyin.tun5,Pinyin.tuo4,Pinyin.tuo4,Pinyin.tuo4,Pinyin.tuo4,Pinyin.wa5,Pinyin.wa5,Pinyin.wa5,Pinyin.wa5,Pinyin.wa5,Pinyin.wai4,Pinyin.wai4,Pinyin.wai4,Pinyin.wan4,Pinyin.wan4,Pinyin.wan4,Pinyin.wan4,Pinyin.wang4,Pinyin.wang4,Pinyin.wang4,Pinyin.wang4,Pinyin.wei4,Pinyin.wei4,Pinyin.wei4,Pinyin.wei4,Pinyin.wen4,Pinyin.wen4,Pinyin.wen4,Pinyin.wen4,Pinyin.weng4,Pinyin.weng4,Pinyin.weng4,Pinyin.wo4,Pinyin.wo4,Pinyin.wo4,Pinyin.wu4,Pinyin.wu4,Pinyin.wu4,Pinyin.wu4,Pinyin.xi4,Pinyin.xi4,Pinyin.xi4,Pinyin.xi4,Pinyin.xia4,Pinyin.xia4,Pinyin.xia4,Pinyin.xia4,Pinyin.xian4,Pinyin.xian4,Pinyin.xian4,Pinyin.xian4,Pinyin.xiang4,Pinyin.xiang4,Pinyin.xiang4,Pinyin.xiang4,Pinyin.xiao4,Pinyin.xiao4,Pinyin.xiao4,Pinyin.xiao4,Pinyin.xie4,Pinyin.xie4,Pinyin.xie4,Pinyin.xie4,Pinyin.xin4,Pinyin.xin4,Pinyin.xin4,Pinyin.xin4,Pinyin.xing4,Pinyin.xing4,Pinyin.xing4,Pinyin.xing4,Pinyin.xiong4,Pinyin.xiong4,Pinyin.xiong4,Pinyin.xiong4,Pinyin.xiu4,Pinyin.xiu4,Pinyin.xiu4,Pinyin.xiu4,Pinyin.xu5,Pinyin.xu5,Pinyin.xu5,Pinyin.xu5,Pinyin.xu5,Pinyin.xuan4,Pinyin.xuan4,Pinyin.xuan4,Pinyin.xuan4,Pinyin.xue4,Pinyin.xue4,Pinyin.xue4,Pinyin.xue4,Pinyin.xun4,Pinyin.xun4,Pinyin.xun4,Pinyin.ya5,Pinyin.ya5,Pinyin.ya5,Pinyin.ya5,Pinyin.ya5,Pinyin.yai2,Pinyin.yan4,Pinyin.yan4,Pinyin.yan4,Pinyin.yan4,Pinyin.yang4,Pinyin.yang4,Pinyin.yang4,Pinyin.yang4,Pinyin.yao4,Pinyin.yao4,Pinyin.yao4,Pinyin.yao4,Pinyin.ye5,Pinyin.ye5,Pinyin.ye5,Pinyin.ye5,Pinyin.ye5,Pinyin.yi5,Pinyin.yi5,Pinyin.yi5,Pinyin.yi5,Pinyin.yi5,Pinyin.yin4,Pinyin.yin4,Pinyin.yin4,Pinyin.yin4,Pinyin.ying4,Pinyin.ying4,Pinyin.ying4,Pinyin.ying4,Pinyin.yo5,Pinyin.yo5,Pinyin.yong4,Pinyin.yong4,Pinyin.yong4,Pinyin.yong4,Pinyin.you4,Pinyin.you4,Pinyin.you4,Pinyin.you4,Pinyin.yu4,Pinyin.yu4,Pinyin.yu4,Pinyin.yu4,Pinyin.yuan4,Pinyin.yuan4,Pinyin.yuan4,Pinyin.yuan4,Pinyin.yue4,Pinyin.yue4,Pinyin.yue4,Pinyin.yun4,Pinyin.yun4,Pinyin.yun4,Pinyin.yun4,Pinyin.za3,Pinyin.za3,Pinyin.za3,Pinyin.zai4,Pinyin.zai4,Pinyin.zai4,Pinyin.zan4,Pinyin.zan4,Pinyin.zan4,Pinyin.zan4,Pinyin.zang4,Pinyin.zang4,Pinyin.zang4,Pinyin.zang4,Pinyin.zao4,Pinyin.zao4,Pinyin.zao4,Pinyin.zao4,Pinyin.ze4,Pinyin.ze4,Pinyin.zei2,Pinyin.zen4,Pinyin.zen4,Pinyin.zen4,Pinyin.zeng4,Pinyin.zeng4,Pinyin.zha4,Pinyin.zha4,Pinyin.zha4,Pinyin.zha4,Pinyin.zhai4,Pinyin.zhai4,Pinyin.zhai4,Pinyin.zhai4,Pinyin.zhan4,Pinyin.zhan4,Pinyin.zhan4,Pinyin.zhan4,Pinyin.zhang4,Pinyin.zhang4,Pinyin.zhang4,Pinyin.zhao4,Pinyin.zhao4,Pinyin.zhao4,Pinyin.zhao4,Pinyin.zhe5,Pinyin.zhe5,Pinyin.zhe5,Pinyin.zhe5,Pinyin.zhe5,Pinyin.zhei4,Pinyin.zhen4,Pinyin.zhen4,Pinyin.zhen4,Pinyin.zheng4,Pinyin.zheng4,Pinyin.zheng4,Pinyin.zhi4,Pinyin.zhi4,Pinyin.zhi4,Pinyin.zhi4,Pinyin.zhong4,Pinyin.zhong4,Pinyin.zhong4,Pinyin.zhou4,Pinyin.zhou4,Pinyin.zhou4,Pinyin.zhou4,Pinyin.zhu4,Pinyin.zhu4,Pinyin.zhu4,Pinyin.zhu4,Pinyin.zhua3,Pinyin.zhua3,Pinyin.zhuai4,Pinyin.zhuai4,Pinyin.zhuai4,Pinyin.zhuan4,Pinyin.zhuan4,Pinyin.zhuan4,Pinyin.zhuang4,Pinyin.zhuang4,Pinyin.zhuang4,Pinyin.zhui4,Pinyin.zhui4,Pinyin.zhui4,Pinyin.zhun4,Pinyin.zhun4,Pinyin.zhun4,Pinyin.zhuo4,Pinyin.zhuo4,Pinyin.zhuo4,Pinyin.zhuo4,Pinyin.zi5,Pinyin.zi5,Pinyin.zi5,Pinyin.zi5,Pinyin.zi5,Pinyin.zong4,Pinyin.zong4,Pinyin.zong4,Pinyin.zou4,Pinyin.zou4,Pinyin.zou4,Pinyin.zu4,Pinyin.zu4,Pinyin.zu4,Pinyin.zu4,Pinyin.zuan4,Pinyin.zuan4,Pinyin.zuan4,Pinyin.zui4,Pinyin.zui4,Pinyin.zui4,Pinyin.zun4,Pinyin.zun4,Pinyin.zun4,Pinyin.zuo5,Pinyin.zuo5,Pinyin.zuo5,Pinyin.zuo5,Pinyin.zuo5,Pinyin.none5,};
    static
    {
        // TODO:什么时候有空了升级到双数组吧
        trie = new Trie().remainLongest();
        map = new TreeMap<String, Pinyin>();
        int end = Pinyin.none5.ordinal();
        for (int i = 0; i < end; ++i)
        {
            Pinyin pinyin = Integer2PinyinConverter.pinyins[i];
            String pinyinWithoutTone = pinyin.getPinyinWithoutTone();
            String firstChar = String.valueOf(pinyin.getFirstChar());
            trie.addKeyword(pinyinWithoutTone);
            trie.addKeyword(firstChar);
            map.put(pinyinWithoutTone, pinyin);
            map.put(firstChar, pinyin);
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
        List<Pinyin> pinyinList = new LinkedList<Pinyin>();
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
        List<Pinyin> pinyinList = new LinkedList<Pinyin>();
        List<Boolean> booleanList = new LinkedList<Boolean>();
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
        return new Pair<List<Pinyin>, List<Boolean>>(pinyinList, booleanList);
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
     * 将拼音的音调统统转为5调或者最大的音调
     * @param p
     * @return
     */
    public static Pinyin convert2Tone5(Pinyin p)
    {
        return tone2tone5[p.ordinal()];
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
            listIterator.set(convert2Tone5(listIterator.next()));
        }

        return pinyinList;
    }
}
