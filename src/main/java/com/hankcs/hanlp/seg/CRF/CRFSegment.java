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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import static com.hankcs.hanlp.utility.Predefine.logger;

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
    protected List<Term> segSentence(String sentence)
    {
        String v[][] = new String[sentence.length()][2];
        int length = sentence.length();
        for (int i = 0; i < length; ++i)
        {
            v[i][0] = String.valueOf(sentence.charAt(i));
        }
        Table table = new Table();
        table.v = v;
        CRFSegmentModel.crfModel.tag(table);
        List<Term> termList = new LinkedList<>();
        StringBuilder sbTerm = new StringBuilder();
        if (HanLP.Config.DEBUG)
        {
            System.out.println("CRF标注结果");
            System.out.println(table);
        }
        for (String[] line : table.v)
        {
            switch (line[1])
            {
                case "B":
                {
                    sbTerm.append(line[0]);
                }break;
                case "E":
                {
                    sbTerm.append(line[0]);
                    termList.add(new Term(sbTerm.toString(), null));
                    sbTerm.setLength(0);
                }break;
                case "M":
                {
                    sbTerm.append(line[0]);
                }break;
                case "S":
                {
                    termList.add(new Term(line[0], null));
                }break;
                default:
                {
                    logger.warning("CRF分词出了意外" + Arrays.toString(line));
                }break;
            }
        }
        if (speechTagging)
        {
            ArrayList<Vertex> vertexList = new ArrayList<>(termList.size() + 1);
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
