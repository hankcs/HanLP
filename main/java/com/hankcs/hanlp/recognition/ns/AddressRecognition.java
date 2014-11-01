/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/07/2014/7/2 12:09</create-date>
 *
 * <copyright file="AddressRecognition.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.recognition.ns;

import com.hankcs.hanlp.dictionary.AddressDictionary;
import com.hankcs.hanlp.dictionary.AddressKeyWordDictionary;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.seg.common.WordNet;
import static com.hankcs.hanlp.utility.Predefine.logger;
import java.util.*;

/**
 * @author hankcs
 */
public class AddressRecognition
{
    public static boolean Recognition(List<Vertex> vertexList, WordNet graphOptimum)
    {
        AddressType[] tagArray = tagByDictionary(vertexList);   // 逆向最长匹配得出标签
        Vertex[] vertexArray = vertexList.toArray(new Vertex[0]);
        List<Vertex> neoVertexList = new LinkedList<Vertex>();
        neoVertexList.add(vertexArray[0]);
        int line = 1;
        for (int i = 1; i <= vertexArray.length - 2; ++i)
        {
            Vertex vertexSc = null;
            AddressType typeSc = null;
            Vertex vertex3rd = null;
            AddressType type3rd = null;
            if (i + 1 <= vertexArray.length - 2)
            {
                vertexSc = vertexArray[i + 1];
                typeSc = tagArray[i + 1];
                if (i + 2 <= vertexArray.length - 2)
                {
                    vertex3rd = vertexArray[i + 2];
                    type3rd = tagArray[i + 2];
                }
            }
            if ((tagArray[i] != null && (maybeInAddress(vertexArray[i + 1], tagArray[i + 1], null, null, null, null))) || maybeInAddress(vertexArray[i], tagArray[i], vertexSc, typeSc, vertex3rd, type3rd))
            {
                Vertex combinedVertex = Vertex.newAddressInstance(vertexArray[i].realWord);
                ++i;
                boolean isBreak = false;
                for (; i <= vertexArray.length - 2; ++i)
                {
                    vertexSc = null;
                    typeSc = null;
                    vertex3rd = null;
                    type3rd = null;
                    if (i + 1 <= vertexArray.length - 2)
                    {
                        vertexSc = vertexArray[i + 1];
                        typeSc = tagArray[i + 1];
                        if (i + 2 <= vertexArray.length - 2)
                        {
                            vertex3rd = vertexArray[i + 2];
                            type3rd = tagArray[i + 2];
                        }
                    }
                    if (maybeInAddress(vertexArray[i], tagArray[i], vertexSc, typeSc, vertex3rd, type3rd))
                    {
                        combinedVertex.realWord += vertexArray[i].realWord;
                    }
                    else
                    {
                        isBreak = true;
                        break;
                    }
                }
                // 应当合并，合并之后加入新的路径中
                neoVertexList.add(combinedVertex);
                // 然后加入词网里
                graphOptimum.add(line, combinedVertex);
                line += combinedVertex.realWord.length();
                if (isBreak)
                {
                    neoVertexList.add(vertexArray[i]);
                    graphOptimum.add(line, vertexArray[i]);
                    line += vertexArray[i].realWord.length();
                }
            }
            else
            {
                neoVertexList.add(vertexArray[i]);
                line += vertexArray[i].realWord.length();
            }
        }
        neoVertexList.add(vertexArray[vertexArray.length - 1]);
        vertexList.clear();
        vertexList.addAll(neoVertexList);
        // 词典分词，加入到词网中
        BaseSearcher searcher = AddressDictionary.getSearcher(graphOptimum.sentence);
        Map.Entry<String, AddressType> entry;
        while ((entry = searcher.next()) != null)
        {
            graphOptimum.add(searcher.getOffset() + 1, Vertex.newAddressInstance(entry.getKey())); // 为了跳过第一行的空格
        }
        graphOptimum.mergeContinuousNsIntoOne();
        return true;
    }

    /**
     * 判断一个顶点是否可能是地址第一部分
     * @param vertexFc 该顶点
     * @param typeFc 该顶点的地址标签
     * @param vertexSc 下一个
     * @param typeSc 下一个
     * @param vertex3rd 第三个
     * @param type3rd 第三个
     * @return 是否可能是
     */
    public static boolean maybeInAddress(Vertex vertexFc, AddressType typeFc, Vertex vertexSc, AddressType typeSc, Vertex vertex3rd, AddressType type3rd)
    {
        return typeFc != null
                || vertexFc.getNature() == Nature.ns
                || ((vertexFc.getNature() == Nature.m || (vertexFc.getNature() == Nature.nx && !vertexFc.realWord.equals("："))) && typeSc != null)
                || (vertexFc.realWord.length() <= 2 && vertexFc.guessNature() != Nature.p && !vertexFc.realWord.equals("：") && typeSc != null && typeSc != AddressType.RelatedPos && typeSc.ordinal() >= AddressType.Town.ordinal())
                || (vertexFc.realWord.length() <= 2 && vertexFc.guessNature() != Nature.p && !vertexFc.realWord.equals("：") && vertexSc != null && vertexSc.realWord.length() <= 2 && vertexSc.guessNature() != Nature.p && !vertexSc.realWord.equals("：") && (type3rd != null && type3rd != AddressType.RelatedPos && type3rd.ordinal() >= AddressType.Town.ordinal()))
                || (vertexFc.realWord.length() == 1 && vertexFc.guessNature() != Nature.p && !vertexFc.realWord.equals("：") && vertexSc != null && vertexSc.realWord.length() == 1 && vertexSc.guessNature() != Nature.p && !vertexSc.realWord.equals("：") && (type3rd != null && type3rd.ordinal() >= AddressType.Town.ordinal()))
        ;
    }

    private static AddressType[] tagByDictionary(List<Vertex> vertexList)
    {
        AddressType[] addressTypeArray = new AddressType[vertexList.size()];
        StringBuilder sbTrace = new StringBuilder("地址标注：");
        int i = 0;
        for (Vertex vertex : vertexList)
        {
            addressTypeArray[i] = AddressDictionary.commonSuffixSearch(vertex.realWord);
            sbTrace.append(vertex.realWord).append('/').append(addressTypeArray[i]).append(',');
            ++i;
        }
        logger.info(sbTrace.toString());
        return addressTypeArray;
    }

    private static int[] tagByKeyWord(List<Vertex> vertexes)
    {
        int[] result = new int[vertexes.size()];
        int i = 0;
        int preTag = 0;
        for (Vertex vertex : vertexes)
        {
            Integer tag = AddressKeyWordDictionary.get(vertex.realWord.substring(vertex.realWord.length() - 1));
            if (vertex.getNature() == Nature.m)
            {
                tag = preTag;
            }
            result[i++] = tag;
            preTag = tag;
        }
        return result;
    }
}
