package com.hankcs.hanlp.seg;

import com.hankcs.hanlp.seg.base.AbstractSegment;
import com.hankcs.hanlp.seg.common.SegmentTypeEnum;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * Date  2019/9/22 17:00
 */
public class Segment {

    private SegmentTypeEnum segmentType;

    private AbstractSegment abstractSegment;

    private Segment(AbstractSegment abstractSegment)
    {
        this.abstractSegment = abstractSegment;
    }

    public void changeSegment(SegmentTypeEnum segmentType)
    {
        this.abstractSegment = segmentType.getSegment();
    }

    public List<Term> seg(String sentence)
    {
        return abstractSegment.seg(sentence);
    }

    static class Builder
    {

        private AbstractSegment abstractSegment;

        public Builder init(SegmentTypeEnum segmentType) {
            abstractSegment = segmentType.getSegment();
            return this;
        }

        public Builder enableCustomDictionaryForcing(boolean enable)
        {
            abstractSegment.enableCustomDictionaryForcing(enable);
            return this;
        }

        public Builder enableCustomDictionary(boolean enable)
        {
            abstractSegment.enableCustomDictionary(enable);
            return this;
        }

        public Builder enableOrganizationRecognize(boolean enable)
        {
            abstractSegment.enableOrganizationRecognize(enable);
            return this;
        }

        public Segment build() {
            return new Segment(abstractSegment);
        }
    }

}
