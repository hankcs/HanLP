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
package com.hankcs.hanlp.seg.NShort.Path;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.utility.Predefine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * 顶点
 * @author hankcs
 */
public class Vertex
{
    static Logger logger = LoggerFactory.getLogger(Vertex.class);
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
     * 在一维顶点数组中的下标，可以视作这个顶点的id
     */
    int index;

    /**
     * 始##始
     */
    public static Vertex B = new Vertex("始##始", " ", new CoreDictionary.Attribute(Nature.begin, Predefine.MAX_FREQUENCY / 10));
    /**
     * 末##末
     */
    public static Vertex E = new Vertex("末##末", " ", new CoreDictionary.Attribute(Nature.begin, Predefine.MAX_FREQUENCY / 10));

    /**
     * 最复杂的构造函数
     * @param word 编译后的词
     * @param realWord 真实词
     * @param attribute 属性
     */
    public Vertex(String word, String realWord, CoreDictionary.Attribute attribute)
    {
        assert word.length() > 0 : "构造空白节点会导致死循环！";
        if (attribute == null) attribute = new CoreDictionary.Attribute(Nature.n, 1);   // 安全起见
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
                    word = Predefine.TAG_PEOPLE;
                }break;
                case ns:
                case nsf:
                {
                    word= Predefine.TAG_PLACE;
                }
                break;
//                case nz:
                case nx:
                {
                        word= Predefine.TAG_PROPER;
                }
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
                {
                    word= Predefine.TAG_GROUP;
                }
                break;
                case m:
                case mq:
                {
                        word= Predefine.TAG_NUMBER;
                }
                break;
                case x:
                {
                    word= Predefine.TAG_CLUSTER;
                }
                break;
                case xx:
                case w:
                {
                    word= Predefine.TAG_OTHER;
                }
                break;
                case t:
                {
                        word= Predefine.TAG_TIME;
                }
                break;
            }
        }
        this.word = word;
        this.realWord = realWord;
        this.attribute = attribute;
    }

    /**
     * 真实词与编译词相同时候的构造函数
     * @param word
     * @param attribute
     */
    public Vertex(String word, CoreDictionary.Attribute attribute)
    {
        this(word, word, attribute);
    }

    /**
     * 通过一个键值对方便地构造节点
     * @param entry
     */
    public Vertex(Map.Entry<String, CoreDictionary.Attribute> entry)
    {
        this(entry.getKey(), entry.getValue());
    }

    /**
     * 自动构造一个合理的顶点
     * @param word
     */
    public Vertex(String word)
    {
        this(word, word, CoreDictionary.GetWordInfo(word));
    }

    public Vertex(char word, CoreDictionary.Attribute attribute)
    {
        this(String.valueOf(word), attribute);
    }

    /**
     * 获取真实词
     * @return
     */
    public String getRealWord()
    {
        return realWord;
    }

    /**
     * 获取词的属性
     * @return
     */
    public CoreDictionary.Attribute getAttribute()
    {
        return attribute;
    }

    /**
     * 与另一个顶点合并
     * @param other 另一个顶点
     * @return 合并后的自己
     */
    public Vertex add(Vertex other)
    {
        this.realWord += other.realWord;
        return this;
    }

    /**
     * 将属性的词性锁定为nature
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
     * @param nature 词性
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
                logger.warn("没有与" + nature + "对应的case");
                break;
        }

        return confirmNature(nature);
    }

    /**
     * 获取该节点的词性，如果词性还未确定，则返回null
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
     * @param realWord 数字对应的真实字串
     * @return 数词顶点
     */
    public static Vertex newNumberInstance(String realWord)
    {
        return new Vertex(Predefine.TAG_NUMBER, realWord, new CoreDictionary.Attribute(Nature.m, 1000));
    }

    /**
     * 创建一个地名实例
     * @param realWord 数字对应的真实字串
     * @return 地名顶点
     */
    public static Vertex newAddressInstance(String realWord)
    {
        return new Vertex(Predefine.TAG_PLACE, realWord, new CoreDictionary.Attribute(Nature.ns, 1000));
    }

    /**
     * 创建一个标点符号实例
     * @param realWord 标点符号对应的真实字串
     * @return 标点符号顶点
     */
    public static Vertex newPunctuationInstance(String realWord)
    {
        return new Vertex(realWord, new CoreDictionary.Attribute(Nature.w, 1000));
    }

    /**
     * 创建一个人名实例
     * @param realWord
     * @return
     */
    public static Vertex newPersonInstance(String realWord)
    {
        return newPersonInstance(realWord, 1000);
    }

    /**
     * 创建一个人名实例
     * @param realWord
     * @param frequency
     * @return
     */
    public static Vertex newPersonInstance(String realWord, int frequency)
    {
        return new Vertex(Predefine.TAG_PEOPLE, realWord, new CoreDictionary.Attribute(Nature.nr, frequency));
    }

    /**
     * 创建一个时间实例
     * @param realWord 时间对应的真实字串
     * @return 时间顶点
     */
    public static Vertex newTimeInstance(String realWord)
    {
        return new Vertex(Predefine.TAG_TIME, realWord, new CoreDictionary.Attribute(Nature.t, 1000));
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
