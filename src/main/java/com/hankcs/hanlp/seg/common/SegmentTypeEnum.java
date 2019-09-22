package com.hankcs.hanlp.seg.common;

import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.HMM.HMMSegment;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;
import com.hankcs.hanlp.seg.base.AbstractSegment;

/**
 * Date  2019/9/22 17:10
 */
public enum SegmentTypeEnum {

    HMM(new HMMSegment()),
    NSHORT(new NShortSegment()),
    VITERBIT(new ViterbiSegment()),
    DIJKSTRA(new DijkstraSegment()),
    DAT(new DoubleArrayTrieSegment());

    private AbstractSegment abstractSegment;

    SegmentTypeEnum(AbstractSegment abstractSegment)
    {
        this.abstractSegment = abstractSegment;
    }

    public AbstractSegment getSegment() {
        return abstractSegment;
    }
}
