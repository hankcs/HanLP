/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 21:18</create-date>
 *
 * <copyright file="TestCRF.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.model;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.model.crf.FeatureTemplate;
import com.hankcs.hanlp.model.crf.CRFModel;
import com.hankcs.hanlp.model.crf.Table;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import junit.framework.TestCase;

import java.io.*;
import java.util.List;

/**
 * @author hankcs
 */
public class TestCRF extends TestCase
{
    public void testTemplate() throws Exception
    {
        FeatureTemplate featureTemplate = FeatureTemplate.create("U05:%x[-2,0]/%x[-1,0]/%x[0,0]");
        Table table = new Table();
        table.v = new String[][]{
                {"那", "S"},
                {"音", "B"},
                {"韵", "E"},};
        char[] parameter = featureTemplate.generateParameter(table, 0);
        System.out.println(parameter);
    }

    public void testTestLoadTemplate() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream("data/test/out.bin"));
        FeatureTemplate featureTemplate = FeatureTemplate.create("U05:%x[-2,0]/%x[-1,0]/%x[0,0]");
        featureTemplate.save(out);
        featureTemplate = new FeatureTemplate();
        featureTemplate.load(ByteArray.createByteArray("data/test/out.bin"));
        System.out.println(featureTemplate);
    }

    public void testLoadFromTxt() throws Exception
    {
        CRFModel model = CRFModel.loadTxt("D:\\Tools\\CRF++-0.58\\example\\seg_cn\\model.txt");
        Table table = new Table();
        table.v = new String[][]{
                {"商", "?"},
                {"品", "?"},
                {"和", "?"},
                {"服", "?"},
                {"务", "?"},
        };
        model.tag(table);
        System.out.println(table);
    }

    public void testLoadModelWhichHasNoB() throws Exception
    {
        CRFModel model = CRFModel.loadTxt("D:\\Tools\\CRF++-0.58\\example\\dependency\\model.txt");
        System.out.println(model);
    }

    public void testSegment() throws Exception
    {
        HanLP.Config.enableDebug();
        CRFSegment segment = new CRFSegment();
//        segment.enablePartOfSpeechTagging(true);
        System.out.println(segment.seg("乐视超级手机能否承载贾布斯的生态梦"));
    }

    /**
     * 现有的CRF效果不满意，重新制作一份以供训练
     *
     * @throws Exception
     */
    public void testPrepareCRFTrainingCorpus() throws Exception
    {
        final BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:\\Tools\\CRF++-0.58\\example\\seg_cn\\2014.txt"), "UTF-8"));
        CorpusLoader.walk("H:\\seg_corpus", new CorpusLoader.Handler()
                          {
                              @Override
                              public void handle(Document document)
                              {
                                  try
                                  {
                                      List<List<IWord>> sentenceList = document.getComplexSentenceList();
                                      for (List<IWord> sentence : sentenceList)
                                      {
                                          for (IWord iWord : sentence)
                                          {
                                              String word = iWord.getValue();
                                              if (word.length() == 1)
                                              {
                                                  bw.write(word);
                                                  bw.write('\t');
                                                  bw.write('S');
                                                  bw.newLine();
                                              }
                                              else
                                              {
                                                  bw.write(word.charAt(0));
                                                  bw.write('\t');
                                                  bw.write('B');
                                                  bw.newLine();
                                                  for (int i = 1; i < word.length() - 1; ++i)
                                                  {
                                                      bw.write(word.charAt(i));
                                                      bw.write('\t');
                                                      bw.write('M');
                                                      bw.newLine();
                                                  }
                                                  bw.write(word.charAt(word.length() - 1));
                                                  bw.write('\t');
                                                  bw.write('E');
                                                  bw.newLine();
                                              }
                                          }
                                          bw.newLine();
                                      }
                                  }
                                  catch (IOException e)
                                  {
                                      e.printStackTrace();
                                  }
                              }
                          }

        );
        bw.close();
    }

    public void testEnglishAndNumber() throws Exception
    {
        String text = "2.34米";
//        System.out.println(CRFSegment.atomSegment(text.toCharArray()));
        HanLP.Config.enableDebug();
        CRFSegment segment = new CRFSegment();
        System.out.println(segment.seg(text));
    }
}
