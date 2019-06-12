package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.seg.common.Vertex;
import java.util.HashMap;
import org.junit.Assert;
import org.junit.Test;

public class MathUtilityTest {
    static final double DELTA = 0.0;

    @Test
    public void testSumInt() {
        Assert.assertEquals(0, MathUtility.sum(new int[0]));
        Assert.assertEquals(106, MathUtility.sum(new int[]{1, 32, 73}));
    }

    @Test
    public void testSumFloat() {
        Assert.assertEquals(0.0f, MathUtility.sum(new float[0]), DELTA);
        Assert.assertEquals(
            22.5f,
            MathUtility.sum(new float[]{1.0f, 5.5f, 16.0f}),
            DELTA
        );
    }

    @Test
    public void testPercentage() {
        Assert.assertEquals(75.0, MathUtility.percentage(96.0, 128.0), DELTA);
        Assert.assertEquals(302.5, MathUtility.percentage(15.125, 5.0), DELTA);
    }

    @Test
    public void testAverage() {
        Assert.assertEquals(
            2.0,
            MathUtility.average(new double[]{1.0, 2.0, 3.0}),
            DELTA
        );
    }

    @Test
    public void testNormalizeExpHashMap() {
        HashMap<String, Double> predictionScores =
            new HashMap<String, Double>();
        predictionScores.put("foo", 1.0);
        predictionScores.put("Bar", 2.0);
        predictionScores.put("test", 0.5);

        HashMap<String, Double> expected = new HashMap<String, Double>();
        expected.put("foo", 0.23122389762214907);
        expected.put("Bar", 0.6285317192117624);
        expected.put("test", 0.14024438316608848);

        MathUtility.normalizeExp(predictionScores);

        Assert.assertEquals(expected, predictionScores);
    }

    @Test
    public void testNormalizeExpDoubleArray() {
        double[] predictionScores = {0, 1, 2, 3};
        double[] expected = {
            0.03205860328008499, 0.08714431874203257,
            0.23688281808991013, 0.6439142598879724
        };
        MathUtility.normalizeExp(predictionScores);

        Assert.assertArrayEquals(expected, predictionScores, DELTA);
    }

    @Test
    public void testCalculateWeight() {
        Nature[] natures = new Nature[]{Nature.begin};
        Vertex vertex1 = new Vertex(
            "Bar",
            new CoreDictionary.Attribute(10),
            55
        );
        Vertex vertex2 = new Vertex(
            "foo",
            new CoreDictionary.Attribute(natures, new int[]{-1}),
            65678
        );

        Assert.assertEquals(2.1972155419463637,
            MathUtility.calculateWeight(vertex1, vertex2),
            DELTA
        );

        Assert.assertEquals(0.10536051123919675,
            MathUtility.calculateWeight(new Vertex("foo"), new Vertex("Bar")),
            DELTA
        );
    }
}
