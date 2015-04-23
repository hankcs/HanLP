/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/10 13:44</create-date>
 *
 * <copyright file="CRFSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.CRF;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algoritm.Viterbi;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionaryTransformMatrixDictionary;
import com.hankcs.hanlp.model.CRFSegmentModel;
import com.hankcs.hanlp.model.crf.Table;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.utility.CharacterHelper;

import java.util.*;


/**
 * 基于CRF的分词器
 *
 * @author hankcs
 */
public class CRFSegment extends Segment
{

    private boolean speechTagging;

    /**
     * 开启词性标注
     *
     * @param enable
     * @return
     */
    public CRFSegment enablePartOfSpeechTagging(boolean enable)
    {
        speechTagging = enable;
        return this;
    }

    @Override
    protected List<Term> segSentence(char[] sentence)
    {
        if (sentence.length == 0) return Collections.emptyList();
        List<String> atomList = atomSegment(sentence);
        Iterator<String> iterator = atomList.iterator();
        int length = atomList.size();
        String v[][] = new String[length][2];
        for (int i = 0; i < length; ++i)
        {
            v[i][0] = iterator.next();
        }
        Table table = new Table();
        table.v = v;
        CRFSegmentModel.crfModel.tag(table);
        List<Term> termList = new LinkedList<Term>();
        if (HanLP.Config.DEBUG)
        {
            System.out.println("CRF标注结果");
            System.out.println(table);
        }
        int offset = 0;
        for (int i = 0; i < table.v.length; offset += table.v[i][0].length(), ++i)
        {
            String[] line = table.v[i];
            switch (line[1].charAt(0))
            {
                case 'B':
                {
                    int begin = offset;
                    while (table.v[i][1].charAt(0) != 'E')
                    {
                        offset += table.v[i][0].length();
                        ++i;
                        if (i == table.v.length)
                        {
                            break;
                        }
                    }
                    if (i == table.v.length)
                    {
                        termList.add(new Term(new String(sentence, begin, offset - begin), null));
                    }
                    else
                        termList.add(new Term(new String(sentence, begin, offset - begin + table.v[i][0].length()), null));
                }
                break;
                default:
                {
                    termList.add(new Term(line[0], null));
                }
                break;
            }
        }
        if (speechTagging)
        {
            ArrayList<Vertex> vertexList = new ArrayList<Vertex>(termList.size() + 1);
            vertexList.add(Vertex.B);
            for (Term term : termList)
            {
                CoreDictionary.Attribute attribute = CoreDictionary.get(term.word);
                if (attribute == null) attribute = new CoreDictionary.Attribute(Nature.nz);
                else term.nature = attribute.nature[0];
                Vertex vertex = new Vertex(term.word, attribute);
                vertexList.add(vertex);
            }
            Viterbi.compute(vertexList, CoreDictionaryTransformMatrixDictionary.transformMatrixDictionary);
            int i = 0;
            for (Term term : termList)
            {
                if (term.nature != null) term.nature = vertexList.get(i + 1).getNature();
                ++i;
            }
        }
        return termList;
    }

    public static List<String> atomSegment(char[] sentence)
    {
        List<String> atomList = new ArrayList<String>(sentence.length);
        final int maxLen = sentence.length - 1;
        final StringBuilder sbAtom = new StringBuilder();
        out:
        for (int i = 0; i < sentence.length; i++)
        {
            if (sentence[i] >= '0' && sentence[i] <= '9')
            {
                sbAtom.append(sentence[i]);
                if (i == maxLen)
                {
                    atomList.add(sbAtom.toString());
                    sbAtom.setLength(0);
                    break;
                }
                char c = sentence[++i];
                while (c == '.' || c == '%' || (c >= '0' && c <= '9'))
                {
                    sbAtom.append(sentence[i]);
                    if (i == maxLen)
                    {
                        atomList.add(sbAtom.toString());
                        sbAtom.setLength(0);
                        break out;
                    }
                    c = sentence[++i];
                }
                atomList.add(sbAtom.toString());
                sbAtom.setLength(0);
                --i;
            }
            else if (CharacterHelper.isEnglishLetter(sentence[i]))
            {
                sbAtom.append(sentence[i]);
                if (i == maxLen)
                {
                    atomList.add(sbAtom.toString());
                    sbAtom.setLength(0);
                    break;
                }
                char c = sentence[++i];
                while (CharacterHelper.isEnglishLetter(c))
                {
                    sbAtom.append(sentence[i]);
                    if (i == maxLen)
                    {
                        atomList.add(sbAtom.toString());
                        sbAtom.setLength(0);
                        break out;
                    }
                    c = sentence[++i];
                }
                atomList.add(sbAtom.toString());
                sbAtom.setLength(0);
                --i;
            }
            else
            {
                atomList.add(String.valueOf(sentence[i]));
            }
        }

        return atomList;
    }
}
