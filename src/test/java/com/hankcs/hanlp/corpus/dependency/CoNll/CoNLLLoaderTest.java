package com.hankcs.hanlp.corpus.dependency.CoNll;

import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Set;

public class CoNLLLoaderTest extends TestCase
{
//    public void testConvert() throws Exception
//    {
//        LinkedList<CoNLLSentence> coNLLSentences = CoNLLLoader.loadSentenceList("D:\\Doc\\语料库\\依存分析训练数据\\THU\\dev.conll.fixed.txt");
//    }
//
//    /**
//     * 细粒度转粗粒度
//     *
//     * @throws Exception
//     */
//    public void testPosTag() throws Exception
//    {
//        DictionaryMaker dictionaryMaker = new DictionaryMaker();
//        LinkedList<CoNLLSentence> coNLLSentences = CoNLLLoader.loadSentenceList("D:\\Doc\\语料库\\依存分析训练数据\\THU\\dev.conll.fixed.txt");
//        for (CoNLLSentence coNLLSentence : coNLLSentences)
//        {
//            for (CoNLLWord coNLLWord : coNLLSentence.word)
//            {
//                dictionaryMaker.add(new Item(coNLLWord.POSTAG, coNLLWord.CPOSTAG));
//            }
//        }
//        System.out.println(dictionaryMaker.entrySet());
//    }
//
//    /**
//     * 导出CRF训练语料
//     *
//     * @throws Exception
//     */
//    public void testMakeCRF() throws Exception
//    {
//        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:\\Tools\\CRF++-0.58\\example\\dependency\\dev.txt")));
//        LinkedList<CoNLLSentence> coNLLSentences = CoNLLLoader.loadSentenceList("D:\\Doc\\语料库\\依存分析训练数据\\THU\\dev.conll.fixed.txt");
//        for (CoNLLSentence coNLLSentence : coNLLSentences)
//        {
//            for (CoNLLWord coNLLWord : coNLLSentence.word)
//            {
//                bw.write(coNLLWord.NAME);
//                bw.write('\t');
//                bw.write(coNLLWord.CPOSTAG);
//                bw.write('\t');
//                bw.write(coNLLWord.POSTAG);
//                bw.write('\t');
//                int d = coNLLWord.HEAD.ID - coNLLWord.ID;
//                int posDistance = 1;
//                if (d > 0)                          // 在后面
//                {
//                    for (int i = 1; i < d; ++i)
//                    {
//                        if (coNLLSentence.word[coNLLWord.ID - 1 + i].CPOSTAG.equals(coNLLWord.HEAD.CPOSTAG))
//                        {
//                            ++posDistance;
//                        }
//                    }
//                }
//                else
//                {
//                    for (int i = 1; i < -d; ++i)    // 在前面
//                    {
//                        if (coNLLSentence.word[coNLLWord.ID - 1 - i].CPOSTAG.equals(coNLLWord.HEAD.CPOSTAG))
//                        {
//                            ++posDistance;
//                        }
//                    }
//                }
//                bw.write((d > 0 ? "+" : "-") + posDistance + "_" + coNLLWord.HEAD.CPOSTAG
////                                 + "_" + coNLLWord.DEPREL
//                );
//                bw.newLine();
//            }
//            bw.newLine();
//        }
//        bw.close();
//    }
//
//    /**
//     * 生成CRF模板
//     *
//     * @throws Exception
//     */
//    public void testMakeCRFTemplate() throws Exception
//    {
//        Set<String> templateList = new LinkedHashSet<String>();
//        int maxDistance = 4;
//        // 字特征
//        for (int i = -maxDistance; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[" + i + ",0]");
//        }
//        // 细词性特征
//        for (int i = -maxDistance; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[" + i + ",1]");
//        }
//        // 粗词性特征
//        for (int i = -maxDistance; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[" + i + ",2]");
//        }
//        // 组合字特征
//        String[] before = new String[maxDistance + 1];
//        String[] after = new String[maxDistance + 1];
//        before[0] = "%x[0,0]";
//        after[0] = "";
//        for (int i = 1; i <= maxDistance; ++i)
//        {
//            before[i] = "%x[-" + i + ",0]/" + before[i - 1];
//            after[i] = after[i - 1] + "/%x[" + i + ",0]";
//        }
//        for (int i = 0; i <= maxDistance; ++i)
//        {
//            for (int j = 0; j <= maxDistance; ++j)
//            {
//                templateList.add(before[i]  + after[j]);
//            }
//        }
//        // 组合粗词性特征
//        before[0] = "%x[0,1]";
//        after[0] = "";
//        for (int i = 1; i <= maxDistance; ++i)
//        {
//            before[i] = "%x[-" + i + ",1]/" + before[i - 1];
//            after[i] = after[i - 1] + "/%x[" + i + ",1]";
//        }
//        for (int i = 0; i <= maxDistance; ++i)
//        {
//            for (int j = 0; j <= maxDistance; ++j)
//            {
//                templateList.add(before[i]  + after[j]);
//            }
//        }
//        // 组合细词性特征
//        before[0] = "%x[0,2]";
//        after[0] = "";
//        for (int i = 1; i <= maxDistance; ++i)
//        {
//            before[i] = "%x[-" + i + ",2]/" + before[i - 1];
//            after[i] = after[i - 1] + "/%x[" + i + ",2]";
//        }
//        for (int i = 0; i <= maxDistance; ++i)
//        {
//            for (int j = 0; j <= maxDistance; ++j)
//            {
//                templateList.add(before[i]  + after[j]);
//            }
//        }
//
//        int id = 0;
//        StringBuilder sb = new StringBuilder();
//        for (String template : templateList)
//        {
//            sb.append(String.format("U%d:%s\n", id, template));
//            ++id;
//        }
//        System.out.println(sb.toString());
//        IOUtil.saveTxt("D:\\Tools\\CRF++-0.58\\example\\dependency\\template.txt", sb);
//    }
//
//    public void testMakeSimpleCRFTemplate() throws Exception
//    {
//        Set<String> templateList = new LinkedHashSet<String>();
//        int maxDistance = 4;
//        // 字特征
//        for (int i = -maxDistance; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[" + i + ",0]");
//        }
//        // 细词性特征
//        for (int i = -maxDistance; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[" + i + ",1]");
//        }
//        // 粗词性特征
//        for (int i = -maxDistance; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[" + i + ",2]");
//        }
//        // 组合特征
//        for (int i = 1; i <= maxDistance; ++i)
//        {
//            templateList.add("%x[-" + i + ",0]/" + "%x[0,0]");
//            templateList.add("%x[0,0]/" + "%x[" + i + ",0]");
//
//            templateList.add("%x[-" + i + ",1]/" + "%x[0,1]");
//            templateList.add("%x[0,1]/" + "%x[" + i + ",1]");
//
//            templateList.add("%x[-" + i + ",2]/" + "%x[0,2]");
//            templateList.add("%x[0,2]/" + "%x[" + i + ",2]");
//        }
//
//        int id = 0;
//        StringBuilder sb = new StringBuilder();
//        for (String template : templateList)
//        {
//            sb.append(String.format("U%d:%s\n", id, template));
//            ++id;
//        }
//        System.out.println(sb.toString());
//        IOUtil.saveTxt("D:\\Tools\\CRF++-0.58\\example\\dependency\\template.txt", sb);
//    }
}