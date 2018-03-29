/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-05 PM7:56</create-date>
 *
 * <copyright file="AveragedPerceptronSegment.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.utility.PosTagUtility;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.seg.CharacterBasedGenerativeModelSegment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.IOException;
import java.util.*;

/**
 * 词法分析器
 *
 * @author hankcs
 */
public class PerceptronLexicalAnalyzer extends CharacterBasedGenerativeModelSegment
{
    private final PerceptronSegmenter segmenter;
    private final PerceptronPOSTagger posTagger;
    private final PerceptionNERecognizer neRecognizer;

    public PerceptronLexicalAnalyzer(LinearModel cwsModel, LinearModel posModel, LinearModel nerModel)
    {
        segmenter = new PerceptronSegmenter(cwsModel);
        if (posModel != null)
        {
            this.posTagger = new PerceptronPOSTagger(posModel);
            config.speechTagging = true;
        }
        else
        {
            this.posTagger = null;
        }
        if (nerModel != null)
        {
            neRecognizer = new PerceptionNERecognizer(nerModel);
            config.ner = true;
        }
        else
        {
            neRecognizer = null;
        }
    }

    public PerceptronLexicalAnalyzer(String cwsModelFile, String posModelFile, String nerModelFile) throws IOException
    {
        this(new LinearModel(cwsModelFile), posModelFile == null ? null : new LinearModel(posModelFile), nerModelFile == null ? null : new LinearModel(nerModelFile));
    }

    public PerceptronLexicalAnalyzer(String cwsModelFile, String posModelFile) throws IOException
    {
        this(new LinearModel(cwsModelFile), posModelFile == null ? null : new LinearModel(posModelFile), null);
    }

    public PerceptronLexicalAnalyzer(String cwsModelFile) throws IOException
    {
        this(new LinearModel(cwsModelFile), null, null);
    }

    public PerceptronLexicalAnalyzer(LinearModel CWSModel)
    {
        this(CWSModel, null, null);
    }

    /**
     * 加载配置文件指定的模型构造词法分析器
     *
     * @throws IOException
     */
    public PerceptronLexicalAnalyzer() throws IOException
    {
        this(HanLP.Config.PerceptronCWSModelPath, HanLP.Config.PerceptronPOSModelPath, HanLP.Config.PerceptronNERModelPath);
    }

    /**
     * 对句子进行词法分析
     *
     * @param sentence 纯文本句子
     * @return HanLP定义的结构化句子
     */
    public Sentence analyze(final String sentence)
    {
        if (sentence.isEmpty())
        {
            return new Sentence(Collections.<IWord>emptyList());
        }
        // 查询词典中的长词
        final List<String> wordList = new LinkedList<String>();
        final int[] offset = new int[]{0};
        if (config.useCustomDictionary)
        {
            CustomDictionary.parseLongestText(sentence, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
            {
                @Override
                public void hit(int begin, int end, CoreDictionary.Attribute value)
                {
                    if (end - begin >= 4 && !value.hasNatureStartsWith("nr") && !value.hasNatureStartsWith("ns") && !value.hasNatureStartsWith("nt")) // 将命名实体识别交给下面去做
                    {
                        if (begin != offset[0])
                        {
                            segmenter.segment(sentence.substring(offset[0], begin), wordList);
                        }
                        wordList.add(sentence.substring(begin, end));
                        offset[0] = end;
                    }
                }
            });
            if (offset[0] != sentence.length())
            {
                segmenter.segment(sentence.substring(offset[0]), wordList);
            }
        }
        else
        {
            segmenter.segment(sentence, wordList);
        }
        String[] wordArray = new String[wordList.size()];
        wordList.toArray(wordArray);
        List<IWord> termList = new ArrayList<IWord>(wordList.size());
        if (posTagger != null)
        {
            String[] posArray = posTagger.tag(wordList);
            if (neRecognizer != null)
            {
                String[] nerArray = neRecognizer.recognize(wordArray, posArray);

                List<Word> result = new LinkedList<Word>();
                result.add(new Word(wordArray[0], posArray[0]));
                String prePos = posArray[0];

                for (int i = 1; i < nerArray.length; i++)
                {
                    if (nerArray[i].charAt(0) == neRecognizer.tagSet.B_TAG_CHAR || nerArray[i].charAt(0) == neRecognizer.tagSet.S_TAG_CHAR || nerArray[i].charAt(0) == neRecognizer.tagSet.O_TAG_CHAR)
                    {
                        termList.add(result.size() > 1 ? new CompoundWord(result, prePos) : result.get(0));
                        result = new ArrayList<Word>();
                    }
                    result.add(new Word(wordArray[i], posArray[i]));
                    if (nerArray[i].charAt(0) == neRecognizer.tagSet.O_TAG_CHAR || nerArray[i].charAt(0) == neRecognizer.tagSet.S_TAG_CHAR)
                    {
                        prePos = posArray[i];
                    }
                    else
                    {
                        prePos = NERTagSet.posOf(nerArray[i]);
                    }
                }
                if (result.size() != 0)
                {
                    termList.add(result.size() > 1 ? new CompoundWord(result, prePos) : result.get(0));
                }
            }
            else
            {
                for (int i = 0; i < wordArray.length; i++)
                {
                    termList.add(new Word(wordArray[i], posArray[i]));
                }
            }
        }
        else
        {
            for (String word : wordArray)
            {
                termList.add(new Word(word, null));
            }
        }

        return new Sentence(termList);
    }


    private void segment(String text, String normalized, List<String> output)
    {
        segmenter.segment(text, normalized, output);
    }

    /**
     * 中文分词
     *
     * @param text
     * @param output
     */
    public void segment(String text, List<String> output)
    {
        String normalized = Utility.normalize(text);
        segment(text, normalized, output);
    }

    /**
     * 中文分词
     *
     * @param sentence
     * @return
     */
    public List<String> segment(String sentence)
    {
        return segmenter.segment(sentence);
    }

    /**
     * 词性标注
     *
     * @param wordList
     * @return
     */
    public String[] partOfSpeechTag(List<String> wordList)
    {
        if (posTagger == null)
        {
            throw new IllegalStateException("未提供词性标注模型");
        }
        return posTagger.tag(wordList);
    }

    /**
     * 命名实体识别
     *
     * @param wordArray
     * @param posArray
     * @return
     */
    public String[] namedEntityRecognize(String[] wordArray, String[] posArray)
    {
        if (neRecognizer == null)
        {
            throw new IllegalStateException("未提供命名实体识别模型");
        }
        return neRecognizer.recognize(wordArray, posArray);
    }

    @Override
    protected List<Term> segSentence(char[] sentence)
    {
        if (sentence.length == 0)
        {
            return Collections.emptyList();
        }
        String original = new String(sentence);
        CharTable.normalization(sentence);
        String normalized = new String(sentence);
        List<String> wordList = new LinkedList<String>();
        segment(original, normalized, wordList);
        List<Term> termList = new ArrayList<Term>(wordList.size());
        int offset = 0;
        for (String word : wordList)
        {
            Term term = new Term(word, null);
            term.offset = offset;
            offset += term.length();
            termList.add(term);
        }
        if (config.speechTagging)
        {
            if (posTagger != null)
            {
                String[] wordArray = new String[wordList.size()];
                wordList.toArray(wordArray);
                String[] posArray = posTagger.tag(wordArray);
                Iterator<Term> iterator = termList.iterator();
                for (String pos : posArray)
                {
                    iterator.next().nature = Nature.create(pos);
                }

                if (config.ner && neRecognizer != null)
                {
                    List<Term> childrenList = null;
                    if (config.isIndexMode())
                    {
                        childrenList = new LinkedList<Term>();
                        iterator = termList.iterator();
                    }
                    termList = new ArrayList<Term>(termList.size());
                    String[] nerArray = neRecognizer.recognize(wordArray, posArray);
                    StringBuilder result = new StringBuilder();
                    result.append(wordArray[0]);
                    if (childrenList != null)
                    {
                        childrenList.add(iterator.next());
                    }
                    String prePos = posArray[0];
                    offset = 0;

                    for (int i = 1; i < nerArray.length; i++)
                    {
                        if (nerArray[i].charAt(0) == neRecognizer.tagSet.B_TAG_CHAR || nerArray[i].charAt(0) == neRecognizer.tagSet.S_TAG_CHAR || nerArray[i].charAt(0) == neRecognizer.tagSet.O_TAG_CHAR)
                        {
                            Term term = new Term(result.toString(), Nature.create(prePos));
                            term.offset = offset;
                            offset += term.length();
                            termList.add(term);
                            if (childrenList != null)
                            {
                                if (childrenList.size() > 1)
                                {
                                    for (Term shortTerm : childrenList)
                                    {
                                        if (shortTerm.length() >= config.indexMode)
                                        {
                                            termList.add(shortTerm);
                                        }
                                    }
                                }
                                childrenList.clear();
                            }
                            result.setLength(0);
                        }
                        result.append(wordArray[i]);
                        if (childrenList != null)
                        {
                            childrenList.add(iterator.next());
                        }
                        if (nerArray[i].charAt(0) == neRecognizer.tagSet.O_TAG_CHAR || nerArray[i].charAt(0) == neRecognizer.tagSet.S_TAG_CHAR)
                        {
                            prePos = posArray[i];
                        }
                        else
                        {
                            prePos = NERTagSet.posOf(nerArray[i]);
                        }
                    }
                    if (result.length() != 0)
                    {
                        Term term = new Term(result.toString(), Nature.create(posArray[posArray.length - 1]));
                        term.offset = offset;
                        termList.add(term);
                        if (childrenList != null)
                        {
                            if (childrenList.size() > 1)
                            {
                                for (Term shortTerm : childrenList)
                                {
                                    if (shortTerm.length() >= config.indexMode)
                                    {
                                        termList.add(shortTerm);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (Term term : termList)
                {
                    CoreDictionary.Attribute attribute = CoreDictionary.get(term.word);
                    if (attribute != null)
                    {
                        term.nature = Nature.create(PosTagUtility.convert(attribute.nature[0]));
                    }
                    else
                    {
                        term.nature = Nature.n;
                    }
                }
            }
        }
        return termList;
    }

    @Override
    protected List<Term> roughSegSentence(char[] sentence)
    {
        return null;
    }

    /**
     * 在线学习
     *
     * @param segmentedTaggedSentence 已分词、标好词性和命名实体的人民日报2014格式的句子
     * @return 是否学习成果（失败的原因是句子格式不合法）
     */
    public boolean learn(String segmentedTaggedSentence)
    {
        Sentence sentence = Sentence.create(segmentedTaggedSentence);
        if (!segmenter.learn(sentence)) return false;
        if (posTagger != null && !posTagger.learn(sentence)) return false;
        if (neRecognizer != null && !neRecognizer.learn(sentence)) return false;
        return true;
    }

    /**
     * 获取分词器
     *
     * @return
     */
    public PerceptronSegmenter getSegmenter()
    {
        return segmenter;
    }

    /**
     * 获取词性标注器
     *
     * @return
     */
    public PerceptronPOSTagger getPOSTagger()
    {
        return posTagger;
    }

    /**
     * 获取命名实体识别器
     *
     * @return
     */
    public PerceptionNERecognizer getNERecognizer()
    {
        return neRecognizer;
    }
}