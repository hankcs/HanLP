package com.hankcs.hanlp.model.perceptron.utility;

import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.model.hmm.HMMNERecognizer;
import com.hankcs.hanlp.model.perceptron.PerceptronNERecognizer;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import junit.framework.TestCase;

import java.util.Arrays;
import java.util.Map;

public class UtilityTest extends TestCase
{
    public void testCombineNER() throws Exception
    {
        NERTagSet nerTagSet = new HMMNERecognizer().getNERTagSet();
        String[] nerArray = Utility.reshapeNER(Utility.convertSentenceToNER(Sentence.create("萨哈夫/nr 说/v ，/w 伊拉克/ns 将/d 同/p [联合国/nt 销毁/v 伊拉克/ns 大规模/b 杀伤性/n 武器/n 特别/a 委员会/n]/nt 继续/v 保持/v 合作/v 。/w"), nerTagSet))[2];
        System.out.println(Arrays.toString(nerArray));
        System.out.println(Utility.combineNER(nerArray, nerTagSet));
    }

    public void testEvaluateNER() throws Exception
    {
        Map<String, double[]> scores = Utility.evaluateNER(new PerceptronNERecognizer(), PKU.PKU199801_TEST);
        Utility.printNERScore(scores);
    }
}