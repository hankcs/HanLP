
package com.hankcs.hanlp;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.io.IIOAdapter;
import com.hankcs.hanlp.corpus.io.ResourceIOAdapter;
import com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser;
import com.hankcs.hanlp.dictionary.py.Pinyin;
import com.hankcs.hanlp.dictionary.py.PinyinDictionary;
import com.hankcs.hanlp.dictionary.ts.HongKongToSimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.HongKongToTaiwanChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.HongKongToTraditionalChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.SimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.SimplifiedToHongKongChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.SimplifiedToTaiwanChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TaiwanToHongKongChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TaiwanToSimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TaiwanToTraditionalChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalToHongKongChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalToTaiwanChineseDictionary;
import com.hankcs.hanlp.phrase.MutualInformationEntropyPhraseExtractor;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.summary.TextRankKeyword;
import com.hankcs.hanlp.summary.TextRankSentence;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;

public class HanLP {
    private HanLP() {
    }

    public static String convertToSimplifiedChinese(String traditionalChineseString) {
        return TraditionalChineseDictionary.convertToSimplifiedChinese(traditionalChineseString.toCharArray());
    }

    public static String convertToTraditionalChinese(String simplifiedChineseString) {
        return SimplifiedChineseDictionary.convertToTraditionalChinese(simplifiedChineseString.toCharArray());
    }

    public static String s2t(String s) {
        return convertToTraditionalChinese(s);
    }

    public static String t2s(String t) {
        return convertToSimplifiedChinese(t);
    }

    public static String s2tw(String s) {
        return SimplifiedToTaiwanChineseDictionary.convertToTraditionalTaiwanChinese(s);
    }

    public static String tw2s(String tw) {
        return TaiwanToSimplifiedChineseDictionary.convertToSimplifiedChinese(tw);
    }

    public static String s2hk(String s) {
        return SimplifiedToHongKongChineseDictionary.convertToTraditionalHongKongChinese(s);
    }

    public static String hk2s(String hk) {
        return HongKongToSimplifiedChineseDictionary.convertToSimplifiedChinese(hk);
    }

    public static String t2tw(String t) {
        return TraditionalToTaiwanChineseDictionary.convertToTaiwanChinese(t);
    }

    public static String tw2t(String tw) {
        return TaiwanToTraditionalChineseDictionary.convertToTraditionalChinese(tw);
    }

    public static String t2hk(String t) {
        return TraditionalToHongKongChineseDictionary.convertToHongKongTraditionalChinese(t);
    }

    public static String hk2t(String hk) {
        return HongKongToTraditionalChineseDictionary.convertToTraditionalChinese(hk);
    }

    public static String hk2tw(String hk) {
        return HongKongToTaiwanChineseDictionary.convertToTraditionalTaiwanChinese(hk);
    }

    public static String tw2hk(String tw) {
        return TaiwanToHongKongChineseDictionary.convertToTraditionalHongKongChinese(tw);
    }

    public static String convertToPinyinString(String text, String separator, boolean remainNone) {
        List pinyinList = PinyinDictionary.convertToPinyin(text, true);
        int length = pinyinList.size();
        StringBuilder sb = new StringBuilder(length * (5 + separator.length()));
        int i = 1;

        for(Iterator var7 = pinyinList.iterator(); var7.hasNext(); ++i) {
            Pinyin pinyin = (Pinyin)var7.next();
            if(pinyin == Pinyin.none5 && !remainNone) {
                sb.append(text.charAt(i - 1));
            } else {
                sb.append(pinyin.getPinyinWithoutTone());
            }

            if(i < length) {
                sb.append(separator);
            }
        }

        return sb.toString();
    }

    public static List<Pinyin> convertToPinyinList(String text) {
        return PinyinDictionary.convertToPinyin(text);
    }

    public static String convertToPinyinFirstCharString(String text, String separator, boolean remainNone) {
        List pinyinList = PinyinDictionary.convertToPinyin(text, remainNone);
        int length = pinyinList.size();
        StringBuilder sb = new StringBuilder(length * (1 + separator.length()));
        int i = 1;

        for(Iterator var7 = pinyinList.iterator(); var7.hasNext(); ++i) {
            Pinyin pinyin = (Pinyin)var7.next();
            sb.append(pinyin.getFirstChar());
            if(i < length) {
                sb.append(separator);
            }
        }

        return sb.toString();
    }

    public static List<Term> segment(String text) {
        return StandardTokenizer.segment(text.toCharArray());
    }

    public static Segment newSegment() {
        return new ViterbiSegment();
    }

    public static CoNLLSentence parseDependency(String sentence) {
        return NeuralNetworkDependencyParser.compute(sentence);
    }

    public static List<String> extractPhrase(String text, int size) {
        MutualInformationEntropyPhraseExtractor extractor = new MutualInformationEntropyPhraseExtractor();
        return extractor.extractPhrase(text, size);
    }

    public static List<String> extractKeyword(String document, int size) {
        return TextRankKeyword.getKeywordList(document, size);
    }

    public static List<String> extractSummary(String document, int size) {
        return TextRankSentence.getTopSentenceList(document, size);
    }

    public static String getSummary(String document, int max_length) {
        return TextRankSentence.getSummary(document, max_length);
    }

    public static final class Config {
        public static boolean DEBUG = false;
        public static String CoreDictionaryPath = "data/dictionary/CoreNatureDictionary.mini.txt";
        public static String CoreDictionaryTransformMatrixDictionaryPath = "data/dictionary/CoreNatureDictionary.tr.txt";
        public static String[] CustomDictionaryPath = new String[]{"data/dictionary/custom/CustomDictionary.txt"};
        public static String BiGramDictionaryPath = "data/dictionary/CoreNatureDictionary.ngram.mini.txt";
        public static String CoreStopWordDictionaryPath = "data/dictionary/stopwords.txt";
        public static String CoreSynonymDictionaryDictionaryPath = "data/dictionary/synonym/CoreSynonym.txt";
        public static String PersonDictionaryPath = "data/dictionary/person/nr.txt";
        public static String PersonDictionaryTrPath = "data/dictionary/person/nr.tr.txt";
        public static String PlaceDictionaryPath = "data/dictionary/place/ns.txt";
        public static String PlaceDictionaryTrPath = "data/dictionary/place/ns.tr.txt";
        public static String OrganizationDictionaryPath = "data/dictionary/organization/nt.txt";
        public static String OrganizationDictionaryTrPath = "data/dictionary/organization/nt.tr.txt";
        public static String tcDictionaryRoot = "data/dictionary/tc/";
        public static String SYTDictionaryPath = "data/dictionary/pinyin/SYTDictionary.txt";
        public static String PinyinDictionaryPath = "data/dictionary/pinyin/pinyin.txt";
        public static String TranslatedPersonDictionaryPath = "data/dictionary/person/nrf.txt";
        public static String JapanesePersonDictionaryPath = "data/dictionary/person/nrj.txt";
        public static String CharTypePath = "data/dictionary/other/CharType.bin";
        public static String CharTablePath = "data/dictionary/other/CharTable.txt";
        public static String WordNatureModelPath = "data/model/dependency/WordNature.txt";
        public static String MaxEntModelPath = "data/model/dependency/MaxEntModel.txt";
        public static String NNParserModelPath = "data/model/dependency/NNParserModel.txt";
        public static String CRFSegmentModelPath = "data/model/segment/CRFSegmentModel.txt";
        public static String HMMSegmentModelPath = "data/model/segment/HMMSegmentModel.bin";
        public static String CRFDependencyModelPath = "data/model/dependency/CRFDependencyModelMini.txt";
        public static boolean ShowTermNature = true;
        public static boolean Normalization = false;
        public static IIOAdapter IOAdapter = new ResourceIOAdapter();

        public Config() {
        }

        public static void enableDebug() {
            enableDebug(true);
        }

        public static void enableDebug(boolean enable) {
            DEBUG = enable;
            if(DEBUG) {
                Predefine.logger.setLevel(Level.ALL);
            } else {
                Predefine.logger.setLevel(Level.OFF);
            }

        }

        static {
            Properties p = new Properties();

            int ioAdapterClassName;
            int e1;
            try {
                ClassLoader e = Thread.currentThread().getContextClassLoader();
                if(e == null) {
                    e = HanLP.Config.class.getClassLoader();
                }

                p.load(new InputStreamReader((InputStream)(Predefine.HANLP_PROPERTIES_PATH == null?e.getResourceAsStream("hanlp.properties"):new FileInputStream(Predefine.HANLP_PROPERTIES_PATH)), "UTF-8"));
                String var14 = p.getProperty("root", "").replaceAll("\\\\", "/");
                if(var14.length() > 0 && !var14.endsWith("/")) {
                    var14 = var14 + '/';
                }

                CoreDictionaryPath = var14 + p.getProperty("CoreDictionaryPath", CoreDictionaryPath);
                CoreDictionaryTransformMatrixDictionaryPath = var14 + p.getProperty("CoreDictionaryTransformMatrixDictionaryPath", CoreDictionaryTransformMatrixDictionaryPath);
                BiGramDictionaryPath = var14 + p.getProperty("BiGramDictionaryPath", BiGramDictionaryPath);
                CoreStopWordDictionaryPath = var14 + p.getProperty("CoreStopWordDictionaryPath", CoreStopWordDictionaryPath);
                CoreSynonymDictionaryDictionaryPath = var14 + p.getProperty("CoreSynonymDictionaryDictionaryPath", CoreSynonymDictionaryDictionaryPath);
                PersonDictionaryPath = var14 + p.getProperty("PersonDictionaryPath", PersonDictionaryPath);
                PersonDictionaryTrPath = var14 + p.getProperty("PersonDictionaryTrPath", PersonDictionaryTrPath);
                String[] var15 = p.getProperty("CustomDictionaryPath", "data/dictionary/custom/CustomDictionary.txt").split(";");
                String var16 = var14;

                for(ioAdapterClassName = 0; ioAdapterClassName < var15.length; ++ioAdapterClassName) {
                    if(var15[ioAdapterClassName].startsWith(" ")) {
                        var15[ioAdapterClassName] = var16 + var15[ioAdapterClassName].trim();
                    } else {
                        var15[ioAdapterClassName] = var14 + var15[ioAdapterClassName];
                        e1 = var15[ioAdapterClassName].lastIndexOf(47);
                        if(e1 != -1) {
                            var16 = var15[ioAdapterClassName].substring(0, e1 + 1);
                        }
                    }
                }

                CustomDictionaryPath = var15;
                tcDictionaryRoot = var14 + p.getProperty("tcDictionaryRoot", tcDictionaryRoot);
                if(!tcDictionaryRoot.endsWith("/")) {
                    tcDictionaryRoot = tcDictionaryRoot + '/';
                }

                SYTDictionaryPath = var14 + p.getProperty("SYTDictionaryPath", SYTDictionaryPath);
                PinyinDictionaryPath = var14 + p.getProperty("PinyinDictionaryPath", PinyinDictionaryPath);
                TranslatedPersonDictionaryPath = var14 + p.getProperty("TranslatedPersonDictionaryPath", TranslatedPersonDictionaryPath);
                JapanesePersonDictionaryPath = var14 + p.getProperty("JapanesePersonDictionaryPath", JapanesePersonDictionaryPath);
                PlaceDictionaryPath = var14 + p.getProperty("PlaceDictionaryPath", PlaceDictionaryPath);
                PlaceDictionaryTrPath = var14 + p.getProperty("PlaceDictionaryTrPath", PlaceDictionaryTrPath);
                OrganizationDictionaryPath = var14 + p.getProperty("OrganizationDictionaryPath", OrganizationDictionaryPath);
                OrganizationDictionaryTrPath = var14 + p.getProperty("OrganizationDictionaryTrPath", OrganizationDictionaryTrPath);
                CharTypePath = var14 + p.getProperty("CharTypePath", CharTypePath);
                CharTablePath = var14 + p.getProperty("CharTablePath", CharTablePath);
                WordNatureModelPath = var14 + p.getProperty("WordNatureModelPath", WordNatureModelPath);
                MaxEntModelPath = var14 + p.getProperty("MaxEntModelPath", MaxEntModelPath);
                NNParserModelPath = var14 + p.getProperty("NNParserModelPath", NNParserModelPath);
                CRFSegmentModelPath = var14 + p.getProperty("CRFSegmentModelPath", CRFSegmentModelPath);
                CRFDependencyModelPath = var14 + p.getProperty("CRFDependencyModelPath", CRFDependencyModelPath);
                HMMSegmentModelPath = var14 + p.getProperty("HMMSegmentModelPath", HMMSegmentModelPath);
                ShowTermNature = "true".equals(p.getProperty("ShowTermNature", "true"));
                Normalization = "true".equals(p.getProperty("Normalization", "false"));
                String var17 = p.getProperty("IOAdapter");
                if(var17 != null) {
                    try {
                        Class var18 = Class.forName(var17);
                        Constructor var19 = var18.getConstructor(new Class[0]);
                        Object instance = var19.newInstance(new Object[0]);
                        if(instance != null) {
                            IOAdapter = (IIOAdapter)instance;
                        }
                    } catch (ClassNotFoundException var9) {
                        Predefine.logger.warning(String.format("找不到IO适配器类： %s ，请检查第三方插件jar包", new Object[]{var17}));
                    } catch (NoSuchMethodException var10) {
                        Predefine.logger.warning(String.format("工厂类[%s]没有默认构造方法，不符合要求", new Object[]{var17}));
                    } catch (SecurityException var11) {
                        Predefine.logger.warning(String.format("工厂类[%s]默认构造方法无法访问，不符合要求", new Object[]{var17}));
                    } catch (Exception var12) {
                        Predefine.logger.warning(String.format("工厂类[%s]构造失败：%s\n", new Object[]{var17, TextUtility.exceptionToString(var12)}));
                    }
                }
            } catch (Exception var13) {
                StringBuilder sbInfo = new StringBuilder("========Tips========\n请将hanlp.properties放在下列目录：\n");
                String classPath = (String)System.getProperties().get("java.class.path");
                if(classPath != null) {
                    String[] prePath = classPath.split(File.pathSeparator);
                    ioAdapterClassName = prePath.length;

                    for(e1 = 0; e1 < ioAdapterClassName; ++e1) {
                        String path = prePath[e1];
                        if((new File(path)).isDirectory()) {
                            sbInfo.append(path).append('\n');
                        }
                    }
                }

                sbInfo.append("Web项目则请放到下列目录：\nWebapp/WEB-INF/lib\nWebapp/WEB-INF/classes\nAppserver/lib\nJRE/lib\n");
                sbInfo.append("并且编辑root=PARENT/path/to/your/data\n");
                sbInfo.append("现在HanLP将尝试从jar包内部resource读取data……");
                Predefine.logger.info("hanlp.properties，进入portable模式。若需要自定义HanLP，请按下列提示操作：\n" + sbInfo);
            }

        }
    }
}
