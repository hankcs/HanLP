/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/2 21:17</create-date>
 *
 * <copyright file="PosTagUtil.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.util;

import com.hankcs.hanlp.seg.common.Term;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hankcs
 */
public class PosTagUtil
{
    /**
     * 转为863标注集<br>
     * 863词性标注集，其各个词性含义如下表：

     Tag	Description	Example	Tag	Description	Example
     a	adjective	美丽	ni	organization name	保险公司
     b	other noun-modifier	大型, 西式	nl	location noun	城郊
     c	conjunction	和, 虽然	ns	geographical name	北京
     d	adverb	很	nt	temporal noun	近日, 明代
     e	exclamation	哎	nz	other proper noun	诺贝尔奖
     g	morpheme	茨, 甥	o	onomatopoeia	哗啦
     h	prefix	阿, 伪	p	preposition	在, 把
     i	idiom	百花齐放	q	quantity	个
     j	abbreviation	公检法	r	pronoun	我们
     k	suffix	界, 率	u	auxiliary	的, 地
     m	number	一, 第一	v	verb	跑, 学习
     n	general noun	苹果	wp	punctuation	，。！
     nd	direction noun	右侧	ws	foreign words	CPU
     nh	person name	杜甫, 汤姆	x	non-lexeme	萄, 翱
     * @param termList
     * @return
     */
    public static List<String> to863(List<Term> termList)
    {
        List<String> posTagList = new ArrayList<String>(termList.size());
        for (Term term : termList)
        {
            String posTag = "x";
            switch (term.nature)
            {
                case bg:
                    posTag = "b";
                    break;
                case mg:
                    posTag = "m";
                    break;
                case nl:
                    posTag = "n";
                    break;
                case nx:
                    posTag = "ws";
                    break;
                case qg:
                    posTag = "q";
                    break;
                case ud:
                case uj:
                case uz:
                case ug:
                case ul:
                case uv:
                    posTag = "u";
                    break;
                case yg:
                    posTag = "u";
                    break;
                case zg:
                    posTag = "u";
                    break;
                case n:
                    posTag = "n";
                    break;
                case nr:
                case nrj:
                case nrf:
                case nr1:
                case nr2:
                    posTag = "nh";
                    break;
                case ns:
                case nsf:
                    posTag = "ns";
                    break;
                case nt:
                case ntc:
                case ntcf:
                case ntcb:
                case ntch:
                case nto:
                case ntu:
                case nts:
                case nth:
                    posTag = "ni";
                    break;
                case nh:
                case nhm:
                case nhd:
                case nn:
                case nnt:
                case nnd:
                    posTag = "nz";
                    break;
                case ng:
                    posTag = "n";
                    break;
                case nf:
                    posTag = "n";
                    break;
                case ni:
                    posTag = "n";
                    break;
                case nit:
                case nic:
                case nis:
                    posTag = "nt";
                    break;
                case nm:
                case nmc:
                case nb:
                case nba:
                case nbc:
                case nbp:
                case nz:
                    posTag = "nz";
                    break;
                case g:
                case gm:
                case gp:
                case gc:
                case gb:
                case gbc:
                case gg:
                case gi:
                    posTag = "nz";
                    break;
                case j:
                    posTag = "j";
                    break;
                case i:
                    posTag = "i";
                    break;
                case l:
                    posTag = "i";
                    break;
                case t:
                    posTag = "nt";
                    break;
                case tg:
                    posTag = "nt";
                    break;
                case s:
                    posTag = "nl";
                    break;
                case f:
                    posTag = "nd";
                    break;
                case v:
                case vd:
                case vn:
                case vshi:
                case vyou:
                case vf:
                case vx:
                case vi:
                case vl:
                case vg:
                    posTag = "v";
                    break;
                case a:
                case ad:
                case an:
                case ag:
                case al:
                    posTag = "a";
                    break;
                case b:
                case bl:
                    posTag = "b";
                    break;
                case z:
                    posTag = "u";
                    break;
                case r:
                case rr:
                case rz:
                case rzt:
                case rzs:
                case rzv:
                case ry:
                case ryt:
                case rys:
                case ryv:
                case rg:
                case Rg:
                    posTag = "r";
                    break;
                case m:
                case mq:
                case Mg:
                    posTag = "m";
                    break;
                case q:
                case qv:
                case qt:
                    posTag = "q";
                    break;
                case d:
                case dg:
                case dl:
                    posTag = "d";
                    break;
                case p:
                case pba:
                case pbei:
                    posTag = "p";
                    break;
                case c:
                case cc:
                    posTag = "c";
                    break;
                case u:
                case uzhe:
                case ule:
                case uguo:
                case ude1:
                case ude2:
                case ude3:
                case usuo:
                case udeng:
                case uyy:
                case udh:
                case uls:
                case uzhi:
                case ulian:
                    posTag = "u";
                    break;
                case e:
                    posTag = "e";
                    break;
                case y:
                    posTag = "e";
                    break;
                case o:
                    posTag = "o";
                    break;
                case h:
                    posTag = "h";
                    break;
                case k:
                    posTag = "k";
                    break;
                case x:
                case xx:
                case xu:
                    posTag = "x";
                    break;
                case w:
                case wkz:
                case wky:
                case wyz:
                case wyy:
                case wj:
                case ww:
                case wt:
                case wd:
                case wf:
                case wn:
                case wm:
                case ws:
                case wp:
                case wb:
                case wh:
                    posTag = "wp";
                    break;
            }

            posTagList.add(posTag);
        }

        return posTagList;
    }
}
