package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import com.hankcs.hanlp.utility.Predefine;
import junit.framework.TestCase;

import java.io.*;
import java.util.List;

public class CRFModelTest extends TestCase
{
//    public void testTemplate() throws Exception
//    {
//        FeatureTemplate featureTemplate = FeatureTemplate.create("U05:%x[-2,0]/%x[-1,0]/%x[0,0]");
//        Table table = new Table();
//        table.v = new String[][]{
//            {"那", "S"},
//            {"音", "B"},
//            {"韵", "E"},};
//        char[] parameter = featureTemplate.generateParameter(table, 0);
//        System.out.println(parameter);
//    }

//    public void testTestLoadTemplate() throws Exception
//    {
//        DataOutputStream out = new DataOutputStream(new FileOutputStream("data/test/out.bin"));
//        FeatureTemplate featureTemplate = FeatureTemplate.create("U05:%x[-2,0]/%x[-1,0]/%x[0,0]");
//        featureTemplate.save(out);
//        featureTemplate = new FeatureTemplate();
//        featureTemplate.load(ByteArray.createByteArray("data/test/out.bin"));
//        System.out.println(featureTemplate);
//    }

//    public void testLoadFromTxt() throws Exception
//    {
//        CRFModel model = CRFModel.loadTxt("D:\\Tools\\CRF++-0.58\\example\\seg_cn\\model.txt");
//        Table table = new Table();
//        table.v = new String[][]{
//            {"商", "?"},
//            {"品", "?"},
//            {"和", "?"},
//            {"服", "?"},
//            {"务", "?"},
//        };
//        model.tag(table);
//        System.out.println(table);
//    }

//    public void testSegment() throws Exception
//    {
//        HanLP.Config.enableDebug();
//        CRFSegment segment = new CRFSegment();
////        segment.enablePartOfSpeechTagging(true);
//        System.out.println(segment.seg("乐视超级手机能否承载贾布斯的生态梦"));
//    }

    /**
     * 现有的CRF效果不满意，重新制作一份以供训练
     *
     * @throws Exception
     */
//    public void testPrepareCRFTrainingCorpus() throws Exception
//    {
//        final BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("e:\\2014.txt"), "UTF-8"));
//        CorpusLoader.walk("D:\\Doc\\语料库\\2014_hankcs", new CorpusLoader.Handler()
//                          {
//                              @Override
//                              public void handle(Document document)
//                              {
//                                  try
//                                  {
//                                      List<List<Word>> sentenceList = document.getSimpleSentenceList();
//                                      if (sentenceList.size() == 0) return;
//                                      for (List<Word> sentence : sentenceList)
//                                      {
//                                          if (sentence.size() == 0) continue;
//                                          for (IWord iWord : sentence)
//                                          {
//                                              String word = iWord.getValue();
//                                              String tag = iWord.getLabel();
//                                              String compiledString = compile(tag);
//                                              if (compiledString != null)
//                                              {
//                                                  word = compiledString;
//                                              }
//                                              if (word.length() == 1 || compiledString != null)
//                                              {
//                                                  bw.write(word);
//                                                  bw.write('\t');
//                                                  bw.write('S');
//                                                  bw.write('\n');
//                                              }
//                                              else
//                                              {
//                                                  bw.write(word.charAt(0));
//                                                  bw.write('\t');
//                                                  bw.write('B');
//                                                  bw.write('\n');
//                                                  for (int i = 1; i < word.length() - 1; ++i)
//                                                  {
//                                                      bw.write(word.charAt(i));
//                                                      bw.write('\t');
//                                                      bw.write('M');
//                                                      bw.write('\n');
//                                                  }
//                                                  bw.write(word.charAt(word.length() - 1));
//                                                  bw.write('\t');
//                                                  bw.write('E');
//                                                  bw.write('\n');
//                                              }
//                                          }
//                                          bw.write('\n');
//                                      }
//                                  }
//                                  catch (IOException e)
//                                  {
//                                      e.printStackTrace();
//                                  }
//                              }
//                          }
//
//        );
//        bw.close();
//    }

//    public void testEnglishAndNumber() throws Exception
//    {
//        String text = "2.34米";
////        System.out.println(CRFSegment.atomSegment(text.toCharArray()));
//        HanLP.Config.enableDebug();
//        CRFSegment segment = new CRFSegment();
//        System.out.println(segment.seg(text));
//    }

    public static String compile(String tag)
    {
        if (tag.startsWith("m")) return "M";
        else if (tag.equals("x")) return "W";
        else if (tag.equals("nx")) return "W";
        return null;
    }

    public void testLoadModelWithBiGramFeature() throws Exception
    {
        String path = HanLP.Config.CRFSegmentModelPath + Predefine.BIN_EXT;
        CRFModel model = new CRFModel(new BinTrie<FeatureFunction>());
        model.load(ByteArray.createByteArray(path));

        Table table = new Table();
        String text = "人民生活进一步改善了";
        table.v = new String[text.length()][2];
        for (int i = 0; i < text.length(); i++)
        {
            table.v[i][0] = String.valueOf(text.charAt(i));
        }

        model.tag(table);
//        System.out.println(table);
    }

//    public void testRemoveSpace() throws Exception
//    {
//        String inputPath = "E:\\2014.txt";
//        String outputPath = "E:\\2014f.txt";
//        BufferedReader br = IOUtil.newBufferedReader(inputPath);
//        BufferedWriter bw = IOUtil.newBufferedWriter(outputPath);
//        String line = "";
//        int preLength = 0;
//        while ((line = br.readLine()) != null)
//        {
//            if (preLength == 0 && line.length() == 0) continue;
//            bw.write(line);
//            bw.newLine();
//            preLength = line.length();
//        }
//        bw.close();
//    }
}