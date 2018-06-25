/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/7 18:47</create-date>
 *
 * <copyright file="DictionaryBasedSegment.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.hanlp.seg;

import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.NShort.Path.AtomNode;

import java.util.List;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 基于词典的机械分词器基类
 *
 * @author hankcs
 */
public abstract class DictionaryBasedSegment extends Segment
{
    /**
     * 开启数词和英文识别（与标准意义上的词性标注不同，只是借用这个配置方法，不是真的开启了词性标注。
     * 一般用词典分词的用户不太可能是NLP专业人士，对词性准确率要求不高，所以干脆不为词典分词实现词性标注。）
     *
     * @param enable
     * @return
     */
    public Segment enablePartOfSpeechTagging(boolean enable)
    {
        return super.enablePartOfSpeechTagging(enable);
    }

    /**
     * 词性标注
     *
     * @param charArray   字符数组
     * @param wordNet     词语长度
     * @param natureArray 输出词性
     */
    protected void posTag(char[] charArray, int[] wordNet, Nature[] natureArray)
    {
        if (config.speechTagging)
        {
            for (int i = 0; i < natureArray.length; )
            {
                if (natureArray[i] == null)
                {
                    int j = i + 1;
                    for (; j < natureArray.length; ++j)
                    {
                        if (natureArray[j] != null) break;
                    }
                    List<AtomNode> atomNodeList = quickAtomSegment(charArray, i, j);
                    for (AtomNode atomNode : atomNodeList)
                    {
                        if (atomNode.sWord.length() >= wordNet[i])
                        {
                            wordNet[i] = atomNode.sWord.length();
                            natureArray[i] = atomNode.getNature();
                            i += wordNet[i];
                        }
                    }
                    i = j;
                }
                else
                {
                    ++i;
                }
            }
        }
    }

    @Override
    public Segment enableCustomDictionary(boolean enable)
    {
        if (enable)
        {
            logger.warning("为基于词典的分词器开启用户词典太浪费了，建议直接将所有词典的路径传入构造函数，这样速度更快、内存更省");
        }
        return super.enableCustomDictionary(enable);
    }
}
