/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/19 21:07</create-date>
 *
 * <copyright file="Vertex.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.common;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.MathTools;
import com.hankcs.hanlp.utility.Predefine;

import java.util.Map;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 顶点
 *
 * @author hankcs
 */
public class Vertex
{
    /**
     * 节点对应的词或等效词（如未##数）
     */
    public String word;
    /**
     * 节点对应的真实词，绝对不含##
     */
    public String realWord;
    /**
     * 词的属性，谨慎修改属性内部的数据，因为会影响到字典<br>
     * 如果要修改，应当new一个Attribute
     */
    public CoreDictionary.Attribute attribute;
    /**
     * 等效词ID,也是Attribute的下标
     */
    public int wordID;

    /**
     * 在一维顶点数组中的下标，可以视作这个顶点的id
     */
    public int index;

    /**
     * 始##始
     */
    public static Vertex B = new Vertex(Predefine.TAG_BIGIN, " ", new CoreDictionary.Attribute(Nature.begin, Predefine.MAX_FREQUENCY / 10), CoreDictionary.getWordID(Predefine.TAG_BIGIN));
    /**
     * 末##末
     */
    public static Vertex E = new Vertex(Predefine.TAG_END, " ", new CoreDictionary.Attribute(Nature.begin, Predefine.MAX_FREQUENCY / 10), CoreDictionary.getWordID(Predefine.TAG_END));

    ////////在最短路相关计算中用到的几个变量，之所以放在这里，是为了避免再去生成对象，浪费时间////////
    /**
     * 到该节点的最短路径的前驱节点
     */
    public Vertex from;
    /**
     * 最短路径对应的权重
     */
    public double weight;

    public void updateFrom(Vertex from)
    {
        double weight = from.weight + MathTools.calculateWeight(from, this);
        if (this.from == null || this.weight > weight)
        {
            this.from = from;
            this.weight = weight;
        }
    }

    /**
     * 最复杂的构造函数
     *
     * @param word      编译后的词
     * @param realWord  真实词
     * @param attribute 属性
     */
    public Vertex(String word, String realWord, CoreDictionary.Attribute attribute)
    {
        this(word, realWord, attribute, -attribute.totalFrequency);
    }

    public Vertex(String word, String realWord, CoreDictionary.Attribute attribute, int wordID)
    {
        if (attribute == null) attribute = new CoreDictionary.Attribute(Nature.n, 1);   // 安全起见
        this.wordID = wordID;
        this.attribute = attribute;
        if (word == null) word = compileRealWord(realWord, attribute);
        assert realWord.length() > 0 : "构造空白节点会导致死循环！";
        this.word = word;
        this.realWord = realWord;
    }

    /**
     * 将原词转为等效词串
     * @param realWord 原来的词
     * @param attribute 等效词串
     * @return
     */
    private String compileRealWord(String realWord, CoreDictionary.Attribute attribute)
    {
        if (attribute.nature.length == 1)
        {
            switch (attribute.nature[0])
            {
                case nr:
                case nr1:
                case nr2:
                case nrf:
                case nrj:
                {
                    wordID = CoreDictionary.NR_WORD_ID;
//                    this.attribute = CoreDictionary.get(CoreDictionary.NR_WORD_ID);
                    return Predefine.TAG_PEOPLE;
                }
                case ns:
                case nsf:
                {
                    wordID = CoreDictionary.NS_WORD_ID;
                    // 在地名识别的时候,希望类似"河镇"的词语保持自己的词性,而不是未##地的词性
//                    this.attribute = CoreDictionary.get(CoreDictionary.NS_WORD_ID);
                    return Predefine.TAG_PLACE;
                }
//                case nz:
                case nx:
                {
                    wordID = CoreDictionary.NX_WORD_ID;
                    this.attribute = CoreDictionary.get(CoreDictionary.NX_WORD_ID);
                    return Predefine.TAG_PROPER;
                }
                case nt:
                case ntc:
                case ntcf:
                case ntcb:
                case ntch:
                case nto:
                case ntu:
                case nts:
                case nth:
                case nit:
                {
                    wordID = CoreDictionary.NT_WORD_ID;
//                    this.attribute = CoreDictionary.get(CoreDictionary.NT_WORD_ID);
                    return Predefine.TAG_GROUP;
                }
                case m:
                case mq:
                {
                    wordID = CoreDictionary.M_WORD_ID;
                    this.attribute = CoreDictionary.get(CoreDictionary.M_WORD_ID);
                    return Predefine.TAG_NUMBER;
                }
                case x:
                {
                    wordID = CoreDictionary.X_WORD_ID;
                    this.attribute = CoreDictionary.get(CoreDictionary.X_WORD_ID);
                    return Predefine.TAG_CLUSTER;
                }
//                case xx:
//                case w:
//                {
//                    word= Predefine.TAG_OTHER;
//                }
//                break;
                case t:
                {
                    wordID = CoreDictionary.T_WORD_ID;
                    this.attribute = CoreDictionary.get(CoreDictionary.T_WORD_ID);
                    return Predefine.TAG_TIME;
                }
            }
        }

        return realWord;
    }

    /**
     * 真实词与编译词相同时候的构造函数
     *
     * @param realWord
     * @param attribute
     */
    public Vertex(String realWord, CoreDictionary.Attribute attribute)
    {
        this(null, realWord, attribute);
    }

    public Vertex(String realWord, CoreDictionary.Attribute attribute, int wordID)
    {
        this(null, realWord, attribute, wordID);
    }

    /**
     * 通过一个键值对方便地构造节点
     *
     * @param entry
     */
    public Vertex(Map.Entry<String, CoreDictionary.Attribute> entry)
    {
        this(entry.getKey(), entry.getValue());
    }

    /**
     * 自动构造一个合理的顶点
     *
     * @param realWord
     */
    public Vertex(String realWord)
    {
        this(null, realWord, CoreDictionary.get(realWord));
    }

    public Vertex(char realWord, CoreDictionary.Attribute attribute)
    {
        this(String.valueOf(realWord), attribute);
    }

    /**
     * 获取真实词
     *
     * @return
     */
    public String getRealWord()
    {
        return realWord;
    }

    /**
     * 获取词的属性
     *
     * @return
     */
    public CoreDictionary.Attribute getAttribute()
    {
        return attribute;
    }

    /**
     * 将属性的词性锁定为nature
     *
     * @param nature 词性
     * @return 如果锁定词性在词性列表中，返回真，否则返回假
     */
    public boolean confirmNature(Nature nature)
    {
        if (attribute.nature.length == 1 && attribute.nature[0] == nature)
        {
            return true;
        }
        boolean result = true;
        int frequency = attribute.getNatureFrequency(nature);
        if (frequency == 0)
        {
            frequency = 1000;
            result = false;
        }
        attribute = new CoreDictionary.Attribute(nature, frequency);
        return result;
    }

    /**
     * 将属性的词性锁定为nature，此重载会降低性能
     *
     * @param nature     词性
     * @param updateWord 是否更新预编译字串
     * @return 如果锁定词性在词性列表中，返回真，否则返回假
     */
    public boolean confirmNature(Nature nature, boolean updateWord)
    {
        switch (nature)
        {

            case m:
                word = Predefine.TAG_NUMBER;
                break;
            case t:
                word = Predefine.TAG_TIME;
                break;
            default:
                logger.warning("没有与" + nature + "对应的case");
                break;
        }

        return confirmNature(nature);
    }

    /**
     * 获取该节点的词性，如果词性还未确定，则返回null
     *
     * @return
     */
    public Nature getNature()
    {
        if (attribute.nature.length == 1)
        {
            return attribute.nature[0];
        }

        return null;
    }

    /**
     * 猜测最可能的词性，也就是这个节点的词性中出现频率最大的那一个词性
     *
     * @return
     */
    public Nature guessNature()
    {
        return attribute.nature[0];
    }

    public boolean hasNature(Nature nature)
    {
        return attribute.getNatureFrequency(nature) > 0;
    }

    /**
     * 复制自己
     *
     * @return 自己的备份
     */
    public Vertex copy()
    {
        return new Vertex(word, realWord, attribute);
    }

    public Vertex setWord(String word)
    {
        this.word = word;
        return this;
    }

    public Vertex setRealWord(String realWord)
    {
        this.realWord = realWord;
        return this;
    }

    /**
     * 创建一个数词实例
     *
     * @param realWord 数字对应的真实字串
     * @return 数词顶点
     */
    public static Vertex newNumberInstance(String realWord)
    {
        return new Vertex(Predefine.TAG_NUMBER, realWord, new CoreDictionary.Attribute(Nature.m, 1000));
    }

    /**
     * 创建一个地名实例
     *
     * @param realWord 数字对应的真实字串
     * @return 地名顶点
     */
    public static Vertex newAddressInstance(String realWord)
    {
        return new Vertex(Predefine.TAG_PLACE, realWord, new CoreDictionary.Attribute(Nature.ns, 1000));
    }

    /**
     * 创建一个标点符号实例
     *
     * @param realWord 标点符号对应的真实字串
     * @return 标点符号顶点
     */
    public static Vertex newPunctuationInstance(String realWord)
    {
        return new Vertex(realWord, new CoreDictionary.Attribute(Nature.w, 1000));
    }

    /**
     * 创建一个人名实例
     *
     * @param realWord
     * @return
     */
    public static Vertex newPersonInstance(String realWord)
    {
        return newPersonInstance(realWord, 1000);
    }

    /**
     * 创建一个音译人名实例
     *
     * @param realWord
     * @return
     */
    public static Vertex newTranslatedPersonInstance(String realWord, int frequency)
    {
        return new Vertex(Predefine.TAG_PEOPLE, realWord, new CoreDictionary.Attribute(Nature.nrf, frequency));
    }

    /**
     * 创建一个日本人名实例
     *
     * @param realWord
     * @return
     */
    public static Vertex newJapanesePersonInstance(String realWord, int frequency)
    {
        return new Vertex(Predefine.TAG_PEOPLE, realWord, new CoreDictionary.Attribute(Nature.nrj, frequency));
    }

    /**
     * 创建一个人名实例
     *
     * @param realWord
     * @param frequency
     * @return
     */
    public static Vertex newPersonInstance(String realWord, int frequency)
    {
        return new Vertex(Predefine.TAG_PEOPLE, realWord, new CoreDictionary.Attribute(Nature.nr, frequency));
    }

    /**
     * 创建一个地名实例
     *
     * @param realWord
     * @param frequency
     * @return
     */
    public static Vertex newPlaceInstance(String realWord, int frequency)
    {
        return new Vertex(Predefine.TAG_PLACE, realWord, new CoreDictionary.Attribute(Nature.ns, frequency));
    }

    /**
     * 创建一个机构名实例
     *
     * @param realWord
     * @param frequency
     * @return
     */
    public static Vertex newOrganizationInstance(String realWord, int frequency)
    {
        return new Vertex(Predefine.TAG_GROUP, realWord, new CoreDictionary.Attribute(Nature.nt, frequency));
    }

    /**
     * 创建一个时间实例
     *
     * @param realWord 时间对应的真实字串
     * @return 时间顶点
     */
    public static Vertex newTimeInstance(String realWord)
    {
        return new Vertex(Predefine.TAG_TIME, realWord, new CoreDictionary.Attribute(Nature.t, 1000));
    }

    /**
     * 生成线程安全的起始节点
     * @return
     */
    public static Vertex newB()
    {
        return new Vertex(Predefine.TAG_BIGIN, " ", new CoreDictionary.Attribute(Nature.begin, Predefine.MAX_FREQUENCY / 10), CoreDictionary.getWordID(Predefine.TAG_BIGIN));
    }

    /**
     * 生成线程安全的终止节点
     * @return
     */
    public static Vertex newE()
    {
        return new Vertex(Predefine.TAG_END, " ", new CoreDictionary.Attribute(Nature.end, Predefine.MAX_FREQUENCY / 10), CoreDictionary.getWordID(Predefine.TAG_END));
    }

    @Override
    public String toString()
    {
        return realWord;
//        return "WordNode{" +
//                "word='" + word + '\'' +
//                (word.equals(realWord) ? "" : (", realWord='" + realWord + '\'')) +
////                ", attribute=" + attribute +
//                '}';
    }
}
