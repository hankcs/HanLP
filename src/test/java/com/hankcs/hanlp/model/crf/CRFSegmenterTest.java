package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.crf.crfpp.crf_learn;
import junit.framework.TestCase;

import java.util.List;

public class CRFSegmenterTest extends TestCase
{

    public static final String CWS_MODEL_PATH = HanLP.Config.CRFCWSModelPath;

    public void testTrain() throws Exception
    {
        CRFSegmenter segmenter = new CRFSegmenter(null);
        segmenter.train("data/test/pku98/199801.txt", CWS_MODEL_PATH);
    }

    public void testConvert() throws Exception
    {
        crf_learn.run("-T " + CWS_MODEL_PATH + " " + CWS_MODEL_PATH + ".txt");
    }

    public void testConvertCorpus() throws Exception
    {
        CRFSegmenter segmenter = new CRFSegmenter(null);
        segmenter.convertCorpus("data/test/pku98/199801.txt", "data/test/crf/cws-corpus.tsv");
        segmenter.dumpTemplate("data/test/crf/cws-template.txt");
    }

    public void testLoad() throws Exception
    {
        CRFSegmenter segmenter = new CRFSegmenter("data/test/converted.txt");
        List<String> wordList = segmenter.segment("商品和服务");
        System.out.println(wordList);
    }

    public void testOutput() throws Exception
    {
//        final CRFSegmenter segmenter = new CRFSegmenter(CWS_MODEL_PATH);
//
//        final BufferedWriter bw = IOUtil.newBufferedWriter("data/test/crf/cws/mdat.txt");
//        IOUtility.loadInstance("data/test/pku98/199801.txt", new InstanceHandler()
//        {
//            @Override
//            public boolean process(Sentence instance)
//            {
//                String text = instance.text().replace("0", "").replace("X", "");
//                try
//                {
//                    for (String term : segmenter.segment(text))
//                    {
//
//                        bw.write(term);
//                        bw.write(" ");
//                    }
//                    bw.newLine();
//                }
//                catch (IOException e)
//                {
//                    e.printStackTrace();
//                }
//                return false;
//            }
//        });
//        bw.close();
    }

}