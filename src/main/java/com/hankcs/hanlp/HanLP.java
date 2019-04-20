/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/17 19:02</create-date>
 *
 * <copyright file="HanLP.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.io.IIOAdapter;
import com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser;
import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;
import com.hankcs.hanlp.dictionary.py.Pinyin;
import com.hankcs.hanlp.dictionary.py.PinyinDictionary;
import com.hankcs.hanlp.dictionary.ts.*;
import com.hankcs.hanlp.mining.phrase.IPhraseExtractor;
import com.hankcs.hanlp.mining.phrase.MutualInformationEntropyPhraseExtractor;
import com.hankcs.hanlp.mining.word.NewWordDiscover;
import com.hankcs.hanlp.mining.word.WordInfo;
import com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer;
import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import com.hankcs.hanlp.seg.HMM.HMMSegment;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.summary.TextRankKeyword;
import com.hankcs.hanlp.summary.TextRankSentence;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * HanLP: Han Language Processing <br>
 * 汉语言处理包 <br>
 * 常用接口工具类
 *
 * @author hankcs
 */
public class HanLP
{
    /**
     * 库的全局配置，既可以用代码修改，也可以通过hanlp.properties配置（按照 变量名=值 的形式）
     */
    public static final class Config
    {
        /**
         * 开发模式
         */
        public static boolean DEBUG = false;
        /**
         * 核心词典路径
         */
        public static String CoreDictionaryPath = "data/dictionary/CoreNatureDictionary.txt";
        /**
         * 核心词典词性转移矩阵路径
         */
        public static String CoreDictionaryTransformMatrixDictionaryPath = "data/dictionary/CoreNatureDictionary.tr.txt";
        /**
         * 用户自定义词典路径
         */
        public static String CustomDictionaryPath[] = new String[]{"data/dictionary/custom/CustomDictionary.txt"};
        /**
         * 2元语法词典路径
         */
        public static String BiGramDictionaryPath = "data/dictionary/CoreNatureDictionary.ngram.txt";

        /**
         * 停用词词典路径
         */
        public static String CoreStopWordDictionaryPath = "data/dictionary/stopwords.txt";
        /**
         * 同义词词典路径
         */
        public static String CoreSynonymDictionaryDictionaryPath = "data/dictionary/synonym/CoreSynonym.txt";
        /**
         * 人名词典路径
         */
        public static String PersonDictionaryPath = "data/dictionary/person/nr.txt";
        /**
         * 人名词典转移矩阵路径
         */
        public static String PersonDictionaryTrPath = "data/dictionary/person/nr.tr.txt";
        /**
         * 地名词典路径
         */
        public static String PlaceDictionaryPath = "data/dictionary/place/ns.txt";
        /**
         * 地名词典转移矩阵路径
         */
        public static String PlaceDictionaryTrPath = "data/dictionary/place/ns.tr.txt";
        /**
         * 地名词典路径
         */
        public static String OrganizationDictionaryPath = "data/dictionary/organization/nt.txt";
        /**
         * 地名词典转移矩阵路径
         */
        public static String OrganizationDictionaryTrPath = "data/dictionary/organization/nt.tr.txt";
        /**
         * 简繁转换词典根目录
         */
        public static String tcDictionaryRoot = "data/dictionary/tc/";

        /**
         * 拼音词典路径
         */
        public static String PinyinDictionaryPath = "data/dictionary/pinyin/pinyin.txt";

        /**
         * 音译人名词典
         */
        public static String TranslatedPersonDictionaryPath = "data/dictionary/person/nrf.txt";

        /**
         * 日本人名词典路径
         */
        public static String JapanesePersonDictionaryPath = "data/dictionary/person/nrj.txt";

        /**
         * 字符类型对应表
         */
        public static String CharTypePath = "data/dictionary/other/CharType.bin";

        /**
         * 字符正规化表（全角转半角，繁体转简体）
         */
        public static String CharTablePath = "data/dictionary/other/CharTable.txt";

        /**
         * 词性标注集描述表，用来进行中英映射（对于Nature词性，可直接参考Nature.java中的注释）
         */
        public static String PartOfSpeechTagDictionary = "data/dictionary/other/TagPKU98.csv";

        /**
         * 词-词性-依存关系模型
         */
        public static String WordNatureModelPath = "data/model/dependency/WordNature.txt";

        /**
         * 最大熵-依存关系模型
         * @deprecated 已废弃，请使用{@link KBeamArcEagerDependencyParser}。未来版本将不再发布该模型，并删除配置项
         */
        public static String MaxEntModelPath = "data/model/dependency/MaxEntModel.txt";
        /**
         * 神经网络依存模型路径
         */
        public static String NNParserModelPath = "data/model/dependency/NNParserModel.txt";
        /**
         * 感知机ArcEager依存模型路径
         */
        public static String PerceptronParserModelPath = "data/model/dependency/perceptron.bin";
        /**
         * CRF分词模型
         *
         * @deprecated 已废弃，请使用{@link com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer}。未来版本将不再发布该模型，并删除配置项
         */
        public static String CRFSegmentModelPath = "data/model/segment/CRFSegmentModel.txt";
        /**
         * HMM分词模型
         *
         * @deprecated 已废弃，请使用{@link PerceptronLexicalAnalyzer}
         */
        public static String HMMSegmentModelPath = "data/model/segment/HMMSegmentModel.bin";
        /**
         * CRF分词模型
         */
        public static String CRFCWSModelPath = "data/model/crf/pku199801/cws.txt";
        /**
         * CRF词性标注模型
         */
        public static String CRFPOSModelPath = "data/model/crf/pku199801/pos.txt";
        /**
         * CRF命名实体识别模型
         */
        public static String CRFNERModelPath = "data/model/crf/pku199801/ner.txt";
        /**
         * 感知机分词模型
         */
        public static String PerceptronCWSModelPath = "data/model/perceptron/large/cws.bin";
        /**
         * 感知机词性标注模型
         */
        public static String PerceptronPOSModelPath = "data/model/perceptron/pku1998/pos.bin";
        /**
         * 感知机命名实体识别模型
         */
        public static String PerceptronNERModelPath = "data/model/perceptron/pku1998/ner.bin";
        /**
         * 分词结果是否展示词性
         */
        public static boolean ShowTermNature = true;
        /**
         * 是否执行字符正规化（繁体->简体，全角->半角，大写->小写），切换配置后必须删CustomDictionary.txt.bin缓存
         */
        public static boolean Normalization = false;
        /**
         * IO适配器（默认null，表示从本地文件系统读取），实现com.hankcs.hanlp.corpus.io.IIOAdapter接口
         * 以在不同的平台（Hadoop、Redis等）上运行HanLP
         */
        public static IIOAdapter IOAdapter;

        static
        {
            // 自动读取配置
            Properties p = new Properties();
            try
            {
                ClassLoader loader = Thread.currentThread().getContextClassLoader();
                if (loader == null)
                {  // IKVM (v.0.44.0.5) doesn't set context classloader
                    loader = HanLP.Config.class.getClassLoader();
                }
                try
                {
                    p.load(new InputStreamReader(Predefine.HANLP_PROPERTIES_PATH == null ?
                                                     loader.getResourceAsStream("hanlp.properties") :
                                                     new FileInputStream(Predefine.HANLP_PROPERTIES_PATH)
                        , "UTF-8"));
                }
                catch (Exception e)
                {
                    String HANLP_ROOT = System.getProperty("HANLP_ROOT");
                    if (HANLP_ROOT == null) HANLP_ROOT = System.getenv("HANLP_ROOT");
                    if (HANLP_ROOT != null)
                    {
                        HANLP_ROOT = HANLP_ROOT.trim();
                        p = new Properties();
                        p.setProperty("root", HANLP_ROOT);
                        logger.info("使用环境变量 HANLP_ROOT=" + HANLP_ROOT);
                    }
                    else throw e;
                }
                String root = p.getProperty("root", "").replaceAll("\\\\", "/");
                if (root.length() > 0 && !root.endsWith("/")) root += "/";
                CoreDictionaryPath = root + p.getProperty("CoreDictionaryPath", CoreDictionaryPath);
                CoreDictionaryTransformMatrixDictionaryPath = root + p.getProperty("CoreDictionaryTransformMatrixDictionaryPath", CoreDictionaryTransformMatrixDictionaryPath);
                BiGramDictionaryPath = root + p.getProperty("BiGramDictionaryPath", BiGramDictionaryPath);
                CoreStopWordDictionaryPath = root + p.getProperty("CoreStopWordDictionaryPath", CoreStopWordDictionaryPath);
                CoreSynonymDictionaryDictionaryPath = root + p.getProperty("CoreSynonymDictionaryDictionaryPath", CoreSynonymDictionaryDictionaryPath);
                PersonDictionaryPath = root + p.getProperty("PersonDictionaryPath", PersonDictionaryPath);
                PersonDictionaryTrPath = root + p.getProperty("PersonDictionaryTrPath", PersonDictionaryTrPath);
                String[] pathArray = p.getProperty("CustomDictionaryPath", "data/dictionary/custom/CustomDictionary.txt").split(";");
                String prePath = root;
                for (int i = 0; i < pathArray.length; ++i)
                {
                    if (pathArray[i].startsWith(" "))
                    {
                        pathArray[i] = prePath + pathArray[i].trim();
                    }
                    else
                    {
                        pathArray[i] = root + pathArray[i];
                        int lastSplash = pathArray[i].lastIndexOf('/');
                        if (lastSplash != -1)
                        {
                            prePath = pathArray[i].substring(0, lastSplash + 1);
                        }
                    }
                }
                CustomDictionaryPath = pathArray;
                tcDictionaryRoot = root + p.getProperty("tcDictionaryRoot", tcDictionaryRoot);
                if (!tcDictionaryRoot.endsWith("/")) tcDictionaryRoot += '/';
                PinyinDictionaryPath = root + p.getProperty("PinyinDictionaryPath", PinyinDictionaryPath);
                TranslatedPersonDictionaryPath = root + p.getProperty("TranslatedPersonDictionaryPath", TranslatedPersonDictionaryPath);
                JapanesePersonDictionaryPath = root + p.getProperty("JapanesePersonDictionaryPath", JapanesePersonDictionaryPath);
                PlaceDictionaryPath = root + p.getProperty("PlaceDictionaryPath", PlaceDictionaryPath);
                PlaceDictionaryTrPath = root + p.getProperty("PlaceDictionaryTrPath", PlaceDictionaryTrPath);
                OrganizationDictionaryPath = root + p.getProperty("OrganizationDictionaryPath", OrganizationDictionaryPath);
                OrganizationDictionaryTrPath = root + p.getProperty("OrganizationDictionaryTrPath", OrganizationDictionaryTrPath);
                CharTypePath = root + p.getProperty("CharTypePath", CharTypePath);
                CharTablePath = root + p.getProperty("CharTablePath", CharTablePath);
                PartOfSpeechTagDictionary = root + p.getProperty("PartOfSpeechTagDictionary", PartOfSpeechTagDictionary);
                WordNatureModelPath = root + p.getProperty("WordNatureModelPath", WordNatureModelPath);
                MaxEntModelPath = root + p.getProperty("MaxEntModelPath", MaxEntModelPath);
                NNParserModelPath = root + p.getProperty("NNParserModelPath", NNParserModelPath);
                PerceptronParserModelPath = root + p.getProperty("PerceptronParserModelPath", PerceptronParserModelPath);
                CRFSegmentModelPath = root + p.getProperty("CRFSegmentModelPath", CRFSegmentModelPath);
                HMMSegmentModelPath = root + p.getProperty("HMMSegmentModelPath", HMMSegmentModelPath);
                CRFCWSModelPath = root + p.getProperty("CRFCWSModelPath", CRFCWSModelPath);
                CRFPOSModelPath = root + p.getProperty("CRFPOSModelPath", CRFPOSModelPath);
                CRFNERModelPath = root + p.getProperty("CRFNERModelPath", CRFNERModelPath);
                PerceptronCWSModelPath = root + p.getProperty("PerceptronCWSModelPath", PerceptronCWSModelPath);
                PerceptronPOSModelPath = root + p.getProperty("PerceptronPOSModelPath", PerceptronPOSModelPath);
                PerceptronNERModelPath = root + p.getProperty("PerceptronNERModelPath", PerceptronNERModelPath);
                ShowTermNature = "true".equals(p.getProperty("ShowTermNature", "true"));
                Normalization = "true".equals(p.getProperty("Normalization", "false"));
                String ioAdapterClassName = p.getProperty("IOAdapter");
                if (ioAdapterClassName != null)
                {
                    try
                    {
                        Class<?> clazz = Class.forName(ioAdapterClassName);
                        Constructor<?> ctor = clazz.getConstructor();
                        Object instance = ctor.newInstance();
                        if (instance != null) IOAdapter = (IIOAdapter) instance;
                    }
                    catch (ClassNotFoundException e)
                    {
                        logger.warning(String.format("找不到IO适配器类： %s ，请检查第三方插件jar包", ioAdapterClassName));
                    }
                    catch (NoSuchMethodException e)
                    {
                        logger.warning(String.format("工厂类[%s]没有默认构造方法，不符合要求", ioAdapterClassName));
                    }
                    catch (SecurityException e)
                    {
                        logger.warning(String.format("工厂类[%s]默认构造方法无法访问，不符合要求", ioAdapterClassName));
                    }
                    catch (Exception e)
                    {
                        logger.warning(String.format("工厂类[%s]构造失败：%s\n", ioAdapterClassName, TextUtility.exceptionToString(e)));
                    }
                }
            }
            catch (Exception e)
            {
                if (new File("data/dictionary/CoreNatureDictionary.tr.txt").isFile())
                {
                    logger.info("使用当前目录下的data");
                }
                else
                {
                    StringBuilder sbInfo = new StringBuilder("========Tips========\n请将hanlp.properties放在下列目录：\n"); // 打印一些友好的tips
                    if (new File("src/main/java").isDirectory())
                    {
                        sbInfo.append("src/main/resources");
                    }
                    else
                    {
                        String classPath = (String) System.getProperties().get("java.class.path");
                        if (classPath != null)
                        {
                            for (String path : classPath.split(File.pathSeparator))
                            {
                                if (new File(path).isDirectory())
                                {
                                    sbInfo.append(path).append('\n');
                                }
                            }
                        }
                        sbInfo.append("Web项目则请放到下列目录：\n" +
                                          "Webapp/WEB-INF/lib\n" +
                                          "Webapp/WEB-INF/classes\n" +
                                          "Appserver/lib\n" +
                                          "JRE/lib\n");
                        sbInfo.append("并且编辑root=PARENT/path/to/your/data\n");
                        sbInfo.append("现在HanLP将尝试从").append(System.getProperties().get("user.dir")).append("读取data……");
                    }
                    logger.severe("没有找到hanlp.properties，可能会导致找不到data\n" + sbInfo);
                }
            }
        }

        /**
         * 开启调试模式(会降低性能)
         */
        public static void enableDebug()
        {
            enableDebug(true);
        }

        /**
         * 开启调试模式(会降低性能)
         *
         * @param enable
         */
        public static void enableDebug(boolean enable)
        {
            DEBUG = enable;
            if (DEBUG)
            {
                logger.setLevel(Level.ALL);
            }
            else
            {
                logger.setLevel(Level.OFF);
            }
        }
    }

    /**
     * 工具类，不需要生成实例
     */
    private HanLP()
    {
    }

    /**
     * 繁转简
     *
     * @param traditionalChineseString 繁体中文
     * @return 简体中文
     */
    public static String convertToSimplifiedChinese(String traditionalChineseString)
    {
        return TraditionalChineseDictionary.convertToSimplifiedChinese(traditionalChineseString.toCharArray());
    }

    /**
     * 简转繁
     *
     * @param simplifiedChineseString 简体中文
     * @return 繁体中文
     */
    public static String convertToTraditionalChinese(String simplifiedChineseString)
    {
        return SimplifiedChineseDictionary.convertToTraditionalChinese(simplifiedChineseString.toCharArray());
    }

    /**
     * 简转繁,是{@link com.hankcs.hanlp.HanLP#convertToTraditionalChinese(java.lang.String)}的简称
     *
     * @param s 简体中文
     * @return 繁体中文(大陆标准)
     */
    public static String s2t(String s)
    {
        return HanLP.convertToTraditionalChinese(s);
    }

    /**
     * 繁转简,是{@link HanLP#convertToSimplifiedChinese(String)}的简称
     *
     * @param t 繁体中文(大陆标准)
     * @return 简体中文
     */
    public static String t2s(String t)
    {
        return HanLP.convertToSimplifiedChinese(t);
    }

    /**
     * 簡體到臺灣正體
     *
     * @param s 簡體
     * @return 臺灣正體
     */
    public static String s2tw(String s)
    {
        return SimplifiedToTaiwanChineseDictionary.convertToTraditionalTaiwanChinese(s);
    }

    /**
     * 臺灣正體到簡體
     *
     * @param tw 臺灣正體
     * @return 簡體
     */
    public static String tw2s(String tw)
    {
        return TaiwanToSimplifiedChineseDictionary.convertToSimplifiedChinese(tw);
    }

    /**
     * 簡體到香港繁體
     *
     * @param s 簡體
     * @return 香港繁體
     */
    public static String s2hk(String s)
    {
        return SimplifiedToHongKongChineseDictionary.convertToTraditionalHongKongChinese(s);
    }

    /**
     * 香港繁體到簡體
     *
     * @param hk 香港繁體
     * @return 簡體
     */
    public static String hk2s(String hk)
    {
        return HongKongToSimplifiedChineseDictionary.convertToSimplifiedChinese(hk);
    }

    /**
     * 繁體到臺灣正體
     *
     * @param t 繁體
     * @return 臺灣正體
     */
    public static String t2tw(String t)
    {
        return TraditionalToTaiwanChineseDictionary.convertToTaiwanChinese(t);
    }

    /**
     * 臺灣正體到繁體
     *
     * @param tw 臺灣正體
     * @return 繁體
     */
    public static String tw2t(String tw)
    {
        return TaiwanToTraditionalChineseDictionary.convertToTraditionalChinese(tw);
    }

    /**
     * 繁體到香港繁體
     *
     * @param t 繁體
     * @return 香港繁體
     */
    public static String t2hk(String t)
    {
        return TraditionalToHongKongChineseDictionary.convertToHongKongTraditionalChinese(t);
    }

    /**
     * 香港繁體到繁體
     *
     * @param hk 香港繁體
     * @return 繁體
     */
    public static String hk2t(String hk)
    {
        return HongKongToTraditionalChineseDictionary.convertToTraditionalChinese(hk);
    }

    /**
     * 香港繁體到臺灣正體
     *
     * @param hk 香港繁體
     * @return 臺灣正體
     */
    public static String hk2tw(String hk)
    {
        return HongKongToTaiwanChineseDictionary.convertToTraditionalTaiwanChinese(hk);
    }

    /**
     * 臺灣正體到香港繁體
     *
     * @param tw 臺灣正體
     * @return 香港繁體
     */
    public static String tw2hk(String tw)
    {
        return TaiwanToHongKongChineseDictionary.convertToTraditionalHongKongChinese(tw);
    }

    /**
     * 转化为拼音
     *
     * @param text       文本
     * @param separator  分隔符
     * @param remainNone 有些字没有拼音（如标点），是否保留它们的拼音（true用none表示，false用原字符表示）
     * @return 一个字符串，由[拼音][分隔符][拼音]构成
     */
    public static String convertToPinyinString(String text, String separator, boolean remainNone)
    {
        List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(text, true);
        int length = pinyinList.size();
        StringBuilder sb = new StringBuilder(length * (5 + separator.length()));
        int i = 1;
        for (Pinyin pinyin : pinyinList)
        {

            if (pinyin == Pinyin.none5 && !remainNone)
            {
                sb.append(text.charAt(i - 1));
            }
            else sb.append(pinyin.getPinyinWithoutTone());
            if (i < length)
            {
                sb.append(separator);
            }
            ++i;
        }
        return sb.toString();
    }

    /**
     * 转化为拼音
     *
     * @param text 待解析的文本
     * @return 一个拼音列表
     */
    public static List<Pinyin> convertToPinyinList(String text)
    {
        return PinyinDictionary.convertToPinyin(text);
    }

    /**
     * 转化为拼音（首字母）
     *
     * @param text       文本
     * @param separator  分隔符
     * @param remainNone 有些字没有拼音（如标点），是否保留它们（用none表示）
     * @return 一个字符串，由[首字母][分隔符][首字母]构成
     */
    public static String convertToPinyinFirstCharString(String text, String separator, boolean remainNone)
    {
        List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(text, remainNone);
        int length = pinyinList.size();
        StringBuilder sb = new StringBuilder(length * (1 + separator.length()));
        int i = 1;
        for (Pinyin pinyin : pinyinList)
        {
            sb.append(pinyin.getFirstChar());
            if (i < length)
            {
                sb.append(separator);
            }
            ++i;
        }
        return sb.toString();
    }

    /**
     * 分词
     *
     * @param text 文本
     * @return 切分后的单词
     */
    public static List<Term> segment(String text)
    {
        return StandardTokenizer.segment(text.toCharArray());
    }

    /**
     * 创建一个分词器<br>
     * 这是一个工厂方法<br>
     * 与直接new一个分词器相比，使用本方法的好处是，以后HanLP升级了，总能用上最合适的分词器
     *
     * @return 一个分词器
     */
    public static Segment newSegment()
    {
        return new ViterbiSegment();   // Viterbi分词器是目前效率和效果的最佳平衡
    }

    /**
     * 创建一个分词器，
     * 这是一个工厂方法<br>
     *
     * @param algorithm 分词算法，传入算法的中英文名都可以，可选列表：<br>
     *                  <ul>
     *                  <li>维特比 (viterbi)：效率和效果的最佳平衡</li>
     *                  <li>双数组trie树 (dat)：极速词典分词，千万字符每秒</li>
     *                  <li>条件随机场 (crf)：分词、词性标注与命名实体识别精度都较高，适合要求较高的NLP任务</li>
     *                  <li>感知机 (perceptron)：分词、词性标注与命名实体识别，支持在线学习</li>
     *                  <li>N最短路 (nshort)：命名实体识别稍微好一些，牺牲了速度</li>
     *                  </ul>
     * @return 一个分词器
     */
    public static Segment newSegment(String algorithm)
    {
        if (algorithm == null)
        {
            throw new IllegalArgumentException(String.format("非法参数 algorithm == %s", algorithm));
        }
        algorithm = algorithm.toLowerCase();
        if ("viterbi".equals(algorithm) || "维特比".equals(algorithm))
            return new ViterbiSegment();   // Viterbi分词器是目前效率和效果的最佳平衡
        else if ("dat".equals(algorithm) || "双数组trie树".equals(algorithm))
            return new DoubleArrayTrieSegment();
        else if ("nshort".equals(algorithm) || "n最短路".equals(algorithm))
            return new NShortSegment();
        else if ("crf".equals(algorithm) || "条件随机场".equals(algorithm))
            try
            {
                return new CRFLexicalAnalyzer();
            }
            catch (IOException e)
            {
                logger.warning("CRF模型加载失败");
                throw new RuntimeException(e);
            }
        else if ("perceptron".equals(algorithm) || "感知机".equals(algorithm))
        {
            try
            {
                return new PerceptronLexicalAnalyzer();
            }
            catch (IOException e)
            {
                logger.warning("感知机模型加载失败");
                throw new RuntimeException(e);
            }
        }
        throw new IllegalArgumentException(String.format("非法参数 algorithm == %s", algorithm));
    }

    /**
     * 依存文法分析
     *
     * @param sentence 待分析的句子
     * @return CoNLL格式的依存关系树
     */
    public static CoNLLSentence parseDependency(String sentence)
    {
        return NeuralNetworkDependencyParser.compute(sentence);
    }

    /**
     * 提取短语
     *
     * @param text 文本
     * @param size 需要多少个短语
     * @return 一个短语列表，大小 <= size
     */
    public static List<String> extractPhrase(String text, int size)
    {
        IPhraseExtractor extractor = new MutualInformationEntropyPhraseExtractor();
        return extractor.extractPhrase(text, size);
    }

    /**
     * 提取词语
     *
     * @param text 大文本
     * @param size 需要提取词语的数量
     * @return 一个词语列表
     */
    public static List<WordInfo> extractWords(String text, int size)
    {
        return extractWords(text, size, false);
    }

    /**
     * 提取词语
     *
     * @param reader 从reader获取文本
     * @param size   需要提取词语的数量
     * @return 一个词语列表
     */
    public static List<WordInfo> extractWords(BufferedReader reader, int size) throws IOException
    {
        return extractWords(reader, size, false);
    }

    /**
     * 提取词语（新词发现）
     *
     * @param text         大文本
     * @param size         需要提取词语的数量
     * @param newWordsOnly 是否只提取词典中没有的词语
     * @return 一个词语列表
     */
    public static List<WordInfo> extractWords(String text, int size, boolean newWordsOnly)
    {
        NewWordDiscover discover = new NewWordDiscover(4, 0.0f, .5f, 100f, newWordsOnly);
        return discover.discover(text, size);
    }

    /**
     * 提取词语（新词发现）
     *
     * @param reader       从reader获取文本
     * @param size         需要提取词语的数量
     * @param newWordsOnly 是否只提取词典中没有的词语
     * @return 一个词语列表
     */
    public static List<WordInfo> extractWords(BufferedReader reader, int size, boolean newWordsOnly) throws IOException
    {
        NewWordDiscover discover = new NewWordDiscover(4, 0.0f, .5f, 100f, newWordsOnly);
        return discover.discover(reader, size);
    }

    /**
     * 提取词语（新词发现）
     *
     * @param reader          从reader获取文本
     * @param size            需要提取词语的数量
     * @param newWordsOnly    是否只提取词典中没有的词语
     * @param max_word_len    词语最长长度
     * @param min_freq        词语最低频率
     * @param min_entropy     词语最低熵
     * @param min_aggregation 词语最低互信息
     * @return 一个词语列表
     */
    public static List<WordInfo> extractWords(BufferedReader reader, int size, boolean newWordsOnly, int max_word_len, float min_freq, float min_entropy, float min_aggregation) throws IOException
    {
        NewWordDiscover discover = new NewWordDiscover(max_word_len, min_freq, min_entropy, min_aggregation, newWordsOnly);
        return discover.discover(reader, size);
    }

    /**
     * 提取关键词
     *
     * @param document 文档内容
     * @param size     希望提取几个关键词
     * @return 一个列表
     */
    public static List<String> extractKeyword(String document, int size)
    {
        return TextRankKeyword.getKeywordList(document, size);
    }

    /**
     * 自动摘要
     * 分割目标文档时的默认句子分割符为，,。:：“”？?！!；;
     *
     * @param document 目标文档
     * @param size     需要的关键句的个数
     * @return 关键句列表
     */
    public static List<String> extractSummary(String document, int size)
    {
        return TextRankSentence.getTopSentenceList(document, size);
    }

    /**
     * 自动摘要
     * 分割目标文档时的默认句子分割符为，,。:：“”？?！!；;
     *
     * @param document   目标文档
     * @param max_length 需要摘要的长度
     * @return 摘要文本
     */
    public static String getSummary(String document, int max_length)
    {
        // Parameter size in this method refers to the string length of the summary required;
        // The actual length of the summary generated may be short than the required length, but never longer;
        return TextRankSentence.getSummary(document, max_length);
    }

    /**
     * 自动摘要
     *
     * @param document           目标文档
     * @param size               需要的关键句的个数
     * @param sentence_separator 分割目标文档时的句子分割符，正则格式， 如：[。？?！!；;]
     * @return 关键句列表
     */
    public static List<String> extractSummary(String document, int size, String sentence_separator)
    {
        return TextRankSentence.getTopSentenceList(document, size, sentence_separator);
    }

    /**
     * 自动摘要
     *
     * @param document           目标文档
     * @param max_length         需要摘要的长度
     * @param sentence_separator 分割目标文档时的句子分割符，正则格式， 如：[。？?！!；;]
     * @return 摘要文本
     */
    public static String getSummary(String document, int max_length, String sentence_separator)
    {
        // Parameter size in this method refers to the string length of the summary required;
        // The actual length of the summary generated may be short than the required length, but never longer;
        return TextRankSentence.getSummary(document, max_length, sentence_separator);
    }

}
