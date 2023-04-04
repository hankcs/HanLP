/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2023-04-04 1:06 PM</create-date>
 *
 * <copyright file="DemoNRF.java">
 * Copyright (c) 2023, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Segment;

import java.io.BufferedWriter;
import java.io.IOException;

/**
 * @author hankcs
 */
public class DemoNRF
{
    public static void main(String[] args) throws IOException
    {
        HanLP.Config.enableDebug();
        String sentence = "我知道卡利斯勒出生于英格兰";
        Segment segment = new DijkstraSegment().enableTranslatedNameRecognize(true);
        System.out.println(segment.seg(sentence));

        if (CoreBiGramTableDictionary.getBiFrequency("未##人", "出生于") == 0)
        {
            BufferedWriter bw = IOUtil.newBufferedWriter(HanLP.Config.BiGramDictionaryPath, true);
            bw.write("\n未##人@出生于 1\n");
            bw.close();
            CoreBiGramTableDictionary.reload();
            System.out.println(segment.seg(sentence));
        }
    }
}
