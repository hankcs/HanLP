package com.hankcs.hanlp.algorithm;

import com.hankcs.hanlp.corpus.synonym.Synonym;
import com.hankcs.hanlp.dictionary.common.CommonSynonymDictionary.SynonymItem;
import org.junit.Assert;
import org.junit.Test;
import java.util.ArrayList;

public class EditDistanceTest {

    @Test
    public void testComputeCharArray() {
        Assert.assertEquals(2,
            EditDistance.compute("foo".toCharArray(), "oof".toCharArray()));
    }

    @Test
    public void testComputeString() {
        Assert.assertEquals(2, EditDistance.compute("foo", "oof"));
    }

    @Test
    public void testComputeList() {
        ArrayList<SynonymItem> synonymItems1 = new ArrayList<SynonymItem>();
        synonymItems1.add(new SynonymItem(new Synonym("", 32L), null, '='));
        
        ArrayList<SynonymItem> synonymItems2 = new ArrayList<SynonymItem>();
        synonymItems2.add(new SynonymItem(new Synonym("", 64L), null, '='));

        Assert.assertEquals(32L,
            EditDistance.compute(synonymItems1, synonymItems2));
    }

    @Test
    public void testComputeLong() {
        Assert.assertEquals(3074457345618258602L,
            EditDistance.compute(new long[]{}, new long[]{4L, 0L}));
        Assert.assertEquals(-15,
            EditDistance.compute(new long[]{-16}, new long[]{32}));
        Assert.assertEquals(0,
            EditDistance.compute(new long[]{16}, new long[]{16}));
    }

    @Test
    public void testComputeInt() {
        Assert.assertEquals(715827882,
            EditDistance.compute(new int[]{4, 0}, new int[]{}));
        Assert.assertEquals(6,
            EditDistance.compute(new int[]{4, 0}, new int[]{8, 16}));
        Assert.assertEquals(0,
            EditDistance.compute(new int[]{16}, new int[]{16}));
    }
}
