package com.hankcs.test.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.util.List;

public class TestTerm extends TestCase {

    public void testContains(){
        List<Term> t1 = HanLP.segment("我在天安门广场吃炸鸡");
        List<Term> t2 = HanLP.segment("我在天安门广场喝啤酒");
        for (Term term:t2)
        {
            if (!t1.contains(term))
            {
                t1.add(term);
            }
        }
        System.out.println(t1);
    }
}
