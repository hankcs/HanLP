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

import java.util.*;


/**
 * 基于CRF的分词器
 * @author hankcs
 */
public class CRFSegment extends Segment
{

    private boolean speechTagging;

    /**
     * 开启词性标注
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
        String v[][] = new String[sentence.length][2];
        int length = sentence.length;
        for (int i = 0; i < length; ++i)
        {
            v[i][0] = String.valueOf(sentence[i]);
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
        for (int i = 0; i < table.v.length; i++)
        {
            String[] line = table.v[i];
            switch (line[1].charAt(0))
            {
                case 'B':
                {
                    int begin = i;
                    while (table.v[i][1].charAt(0) != 'E')
                    {
                        ++i;
                        if (i == table.v.length)
                        {
                            break;
                        }
                    }
                    if (i == table.v.length)
                    {
                        termList.add(new Term(new String(sentence, begin, i - begin), null));
                    }
                    else
                        termList.add(new Term(new String(sentence, begin, i - begin + 1), null));
                }break;
                default:
                {
                    termList.add(new Term(line[0], null));
                }break;
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
}
