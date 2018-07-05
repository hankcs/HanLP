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

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.perceptron.PerceptronTrainer;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.utility.IOUtility;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.lexical.POSTagger;

import java.util.*;

/**
 * @author hankcs
 */
public class PosTagUtil
{
    private static Map<String, String> posConverter = new TreeMap<String, String>();

    static
    {
        posConverter.put("Mg", "m");
        posConverter.put("Rg", "r");
        posConverter.put("ad", "a");
        posConverter.put("ag", "a");
        posConverter.put("al", "a");
        posConverter.put("an", "a");
        posConverter.put("begin", "x");
        posConverter.put("bg", "b");
        posConverter.put("bl", "b");
        posConverter.put("cc", "c");
        posConverter.put("dg", "d");
        posConverter.put("dl", "d");
        posConverter.put("end", "x");
        posConverter.put("f", "nd");
        posConverter.put("g", "nz");
        posConverter.put("gb", "nz");
        posConverter.put("gbc", "nz");
        posConverter.put("gc", "nz");
        posConverter.put("gg", "nz");
        posConverter.put("gi", "nz");
        posConverter.put("gm", "nz");
        posConverter.put("gp", "nz");
        posConverter.put("l", "i");
        posConverter.put("mg", "m");
        posConverter.put("mq", "m");
        posConverter.put("nb", "nz");
        posConverter.put("nba", "nz");
        posConverter.put("nbc", "nz");
        posConverter.put("nbp", "nz");
        posConverter.put("nf", "n");
        posConverter.put("ng", "n");
        posConverter.put("nh", "nz");
        posConverter.put("nhd", "nz");
        posConverter.put("nhm", "nz");
        posConverter.put("ni", "n");
        posConverter.put("nic", "nt");
        posConverter.put("nis", "nt");
        posConverter.put("nit", "nt");
        posConverter.put("nl", "n");
        posConverter.put("nm", "nz");
        posConverter.put("nmc", "nz");
        posConverter.put("nn", "nz");
        posConverter.put("nnd", "nz");
        posConverter.put("nnt", "nz");
        posConverter.put("nr", "nh");
        posConverter.put("nr1", "nh");
        posConverter.put("nr2", "nh");
        posConverter.put("nrf", "nh");
        posConverter.put("nrj", "nh");
        posConverter.put("nsf", "ns");
        posConverter.put("nt", "ni");
        posConverter.put("ntc", "ni");
        posConverter.put("ntcb", "ni");
        posConverter.put("ntcf", "ni");
        posConverter.put("ntch", "ni");
        posConverter.put("nth", "ni");
        posConverter.put("nto", "ni");
        posConverter.put("nts", "ni");
        posConverter.put("ntu", "ni");
        posConverter.put("nx", "ws");
        posConverter.put("pba", "p");
        posConverter.put("pbei", "p");
        posConverter.put("qg", "q");
        posConverter.put("qt", "q");
        posConverter.put("qv", "q");
        posConverter.put("rg", "r");
        posConverter.put("rr", "r");
        posConverter.put("ry", "r");
        posConverter.put("rys", "r");
        posConverter.put("ryt", "r");
        posConverter.put("ryv", "r");
        posConverter.put("rz", "r");
        posConverter.put("rzs", "r");
        posConverter.put("rzt", "r");
        posConverter.put("rzv", "r");
        posConverter.put("s", "nl");
        posConverter.put("t", "nt");
        posConverter.put("tg", "nt");
        posConverter.put("ud", "u");
        posConverter.put("ude1", "u");
        posConverter.put("ude2", "u");
        posConverter.put("ude3", "u");
        posConverter.put("udeng", "u");
        posConverter.put("udh", "u");
        posConverter.put("ug", "u");
        posConverter.put("uguo", "u");
        posConverter.put("uj", "u");
        posConverter.put("ul", "u");
        posConverter.put("ule", "u");
        posConverter.put("ulian", "u");
        posConverter.put("uls", "u");
        posConverter.put("usuo", "u");
        posConverter.put("uv", "u");
        posConverter.put("uyy", "u");
        posConverter.put("uz", "u");
        posConverter.put("uzhe", "u");
        posConverter.put("uzhi", "u");
        posConverter.put("vd", "v");
        posConverter.put("vf", "v");
        posConverter.put("vg", "v");
        posConverter.put("vi", "v");
        posConverter.put("vl", "v");
        posConverter.put("vn", "v");
        posConverter.put("vshi", "v");
        posConverter.put("vx", "v");
        posConverter.put("vyou", "v");
        posConverter.put("w", "wp");
        posConverter.put("wb", "wp");
        posConverter.put("wd", "wp");
        posConverter.put("wf", "wp");
        posConverter.put("wh", "wp");
        posConverter.put("wj", "wp");
        posConverter.put("wky", "wp");
        posConverter.put("wkz", "wp");
        posConverter.put("wm", "wp");
        posConverter.put("wn", "wp");
        posConverter.put("ws", "wp");
        posConverter.put("wt", "wp");
        posConverter.put("ww", "wp");
        posConverter.put("wyy", "wp");
        posConverter.put("wyz", "wp");
        posConverter.put("xu", "x");
        posConverter.put("xx", "x");
        posConverter.put("y", "e");
        posConverter.put("yg", "u");
        posConverter.put("z", "u");
        posConverter.put("zg", "u");
    }

    /**
     * 转为863标注集<br>
     * 863词性标注集，其各个词性含义如下表：
     * <p>
     * Tag	Description	Example	Tag	Description	Example
     * a	adjective	美丽	ni	organization name	保险公司
     * b	other noun-modifier	大型, 西式	nl	location noun	城郊
     * c	conjunction	和, 虽然	ns	geographical name	北京
     * d	adverb	很	nt	temporal noun	近日, 明代
     * e	exclamation	哎	nz	other proper noun	诺贝尔奖
     * g	morpheme	茨, 甥	o	onomatopoeia	哗啦
     * h	prefix	阿, 伪	p	preposition	在, 把
     * i	idiom	百花齐放	q	quantity	个
     * j	abbreviation	公检法	r	pronoun	我们
     * k	suffix	界, 率	u	auxiliary	的, 地
     * m	number	一, 第一	v	verb	跑, 学习
     * n	general noun	苹果	wp	punctuation	，。！
     * nd	direction noun	右侧	ws	foreign words	CPU
     * nh	person name	杜甫, 汤姆	x	non-lexeme	萄, 翱
     *
     * @param termList
     * @return
     */
    public static List<String> to863(List<Term> termList)
    {
        List<String> posTagList = new ArrayList<String>(termList.size());
        for (Term term : termList)
        {
            String posTag = posConverter.get(term.nature.toString());
            if (posTag == null)
                posTag = term.nature.toString();
            posTagList.add(posTag);
        }

        return posTagList;
    }

    /**
     * 评估词性标注器的准确率
     *
     * @param tagger 词性标注器
     * @param corpus 测试集
     * @return Accuracy百分比
     */
    public static float evaluate(POSTagger tagger, String corpus)
    {
        int correct = 0, total = 0;
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(corpus);
        for (String line : lineIterator)
        {
            Sentence sentence = Sentence.create(line);
            if (sentence == null) continue;
            String[][] wordTagArray = sentence.toWordTagArray();
            String[] prediction = tagger.tag(wordTagArray[0]);
            assert prediction.length == wordTagArray[1].length;
            total += prediction.length;
            for (int i = 0; i < prediction.length; i++)
            {
                if (prediction[i].equals(wordTagArray[1][i]))
                    ++correct;
            }
        }
        if (total == 0) return 0;
        return correct / (float) total * 100;
    }
}
