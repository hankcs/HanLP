package com.hankcs.hanlp;

import com.hankcs.hanlp.corpus.occurrence.Occurrence;
import com.hankcs.hanlp.corpus.occurrence.TriaFrequency;

import java.util.Map;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;
import com.hankcs.hanlp.utility.Predefine;

/**
 * Created by chenjianfeng on 2017/7/25.
 */
public class Test {
    public static void main(String[] args) throws Exception{
        ViterbiSegment vs = new ViterbiSegment();
        System.out.println(vs.seg("分词测试样例！"));
    }
}
