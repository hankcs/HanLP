package com.hankcs.hanlp.model.trigram;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.seg.HMM.HMMSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.util.LinkedList;
import java.util.List;

public class CharacterBasedGenerativeModelTest extends TestCase
{
//    public void testTrainAndSegment() throws Exception
//    {
//        final CharacterBasedGenerativeModel model = new CharacterBasedGenerativeModel();
//        CorpusLoader.walk("D:\\JavaProjects\\HanLP\\data\\test\\cbgm", new CorpusLoader.Handler()
//        {
//            @Override
//            public void handle(Document document)
//            {
//                for (List<Word> sentence : document.getSimpleSentenceList())
//                {
//                    model.learn(sentence);
//                }
//            }
//        });
//        model.train();
////        DataOutputStream out = new DataOutputStream(new FileOutputStream(HanLP.Config.HMMSegmentModelPath));
////        model.save(out);
////        out.close();
////        model.load(ByteArray.createByteArray(HanLP.Config.HMMSegmentModelPath));
//        String text = "中国领土";
//        char[] charArray = text.toCharArray();
//        char[] tag = model.tag(charArray);
//        System.out.println(tag);
//    }
//
//    public void testLoad() throws Exception
//    {
//        CharacterBasedGenerativeModel model = new CharacterBasedGenerativeModel();
//        model.load(ByteArray.createByteArray(HanLP.Config.HMMSegmentModelPath));
//        String text = "我实现了一个基于Character Based TriGram的分词器";
//        char[] sentence = text.toCharArray();
//        char[] tag = model.tag(sentence);
//
//        List<String> termList = new LinkedList<String>();
//        int offset = 0;
//        for (int i = 0; i < tag.length; offset += 1, ++i)
//        {
//            switch (tag[i])
//            {
//                case 'b':
//                {
//                    int begin = offset;
//                    while (tag[i] != 'e')
//                    {
//                        offset += 1;
//                        ++i;
//                        if (i == tag.length)
//                        {
//                            break;
//                        }
//                    }
//                    if (i == tag.length)
//                    {
//                        termList.add(new String(sentence, begin, offset - begin));
//                    }
//                    else
//                        termList.add(new String(sentence, begin, offset - begin + 1));
//                }
//                break;
//                default:
//                {
//                    termList.add(new String(sentence, offset, 1));
//                }
//                break;
//            }
//        }
//        System.out.println(tag);
//        System.out.println(termList);
//    }
//
//    public void testSegment() throws Exception
//    {
//        HanLP.Config.ShowTermNature = false;
//        String text = "我实现了一个基于Character Based TriGram的分词器";
//        Segment segment = new HMMSegment();
//        List<Term> termList = segment.seg(text);
//        System.out.println(termList);
//    }
}