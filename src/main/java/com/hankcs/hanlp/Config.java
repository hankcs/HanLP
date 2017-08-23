package com.hankcs.hanlp;

import com.hankcs.hanlp.corpus.io.IIOAdapter;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.util.Properties;
import java.util.logging.Level;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * Created by chenjianfeng on 2017/7/27.
 */
public class Config implements Serializable{
    /**
     * 开发模式
     */
    public static boolean DEBUG = false;
    /**
     * HDFS根路径
     */
    public static String HdfsRoot = "";
    /**
     * 核心词典路径
     */
    public static String CoreDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/CoreNatureDictionary.txt";
    /**
     * 核心词典词性转移矩阵路径
     */
    public static String CoreDictionaryTransformMatrixDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/CoreNatureDictionary.tr.txt";
    /**
     * 用户自定义词典路径
     */
    public static String CustomDictionaryPath[] = new String[]{"hdfs://your/hadoop/abs/path/data/dictionary/custom/CustomDictionary.txt"};
    /**
     * 2元语法词典路径
     */
    public static String BiGramDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/CoreNatureDictionary.ngram.txt";

    /**
     * 停用词词典路径
     */
    public static String CoreStopWordDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/stopwords.txt";
    /**
     * 同义词词典路径
     */
    public static String CoreSynonymDictionaryDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/synonym/CoreSynonym.txt";
    /**
     * 人名词典路径
     */
    public static String PersonDictionaryPath = "hdfs://your/hadoop/abs/path/dictionary/person/nr.txt";
    /**
     * 人名词典转移矩阵路径
     */
    public static String PersonDictionaryTrPath = "hdfs://your/hadoop/abs/path/data/dictionary/person/nr.tr.txt";
    /**
     * 地名词典路径
     */
    public static String PlaceDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/place/ns.txt";
    /**
     * 地名词典转移矩阵路径
     */
    public static String PlaceDictionaryTrPath = "hdfs://your/hadoop/abs/path/data/dictionary/place/ns.tr.txt";
    /**
     * 地名词典路径
     */
    public static String OrganizationDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/organization/nt.txt";
    /**
     * 地名词典转移矩阵路径
     */
    public static String OrganizationDictionaryTrPath = "hdfs://your/hadoop/abs/path/data/dictionary/organization/nt.tr.txt";
    /**
     * 简繁转换词典根目录
     */
    public static String tcDictionaryRoot = "hdfs://your/hadoop/abs/path/data/dictionary/tc/";
    /**
     * 声母韵母语调词典
     */
    public static String SYTDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/pinyin/SYTDictionary.txt";

    /**
     * 拼音词典路径
     */
    public static String PinyinDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/pinyin/pinyin.txt";

    /**
     * 音译人名词典
     */
    public static String TranslatedPersonDictionaryPath = "hdfs://your/hadoop/abs/path/hanlp/data/dictionary/person/nrf.txt";

    /**
     * 日本人名词典路径
     */
    public static String JapanesePersonDictionaryPath = "hdfs://your/hadoop/abs/path/data/dictionary/person/nrj.txt";

    /**
     * 字符类型对应表
     */
    public static String CharTypePath = "hdfs://your/hadoop/abs/path/data/dictionary/other/CharType.bin";

    /**
     * 字符正规化表（全角转半角，繁体转简体）
     */
    public static String CharTablePath = "hdfs://your/hadoop/abs/path/data/dictionary/other/CharTable.txt";

    /**
     * 词-词性-依存关系模型
     */
    public static String WordNatureModelPath = "hdfs://your/hadoop/abs/path/data/model/dependency/WordNature.txt";

    /**
     * 最大熵-依存关系模型
     */
    public static String MaxEntModelPath = "hdfs://your/hadoop/abs/path/data/model/dependency/MaxEntModel.txt";
    /**
     * 神经网络依存模型路径
     */
    public static String NNParserModelPath = "hdfs://your/hadoop/abs/path/data/model/dependency/NNParserModel.txt";
    /**
     * CRF分词模型
     */
    public static String CRFSegmentModelPath = "hdfs://your/hadoop/abs/path/data/model/segment/CRFSegmentModel.txt";
    /**
     * HMM分词模型
     */
    public static String HMMSegmentModelPath = "hdfs://your/hadoop/abs/path/data/model/segment/HMMSegmentModel.bin";
    /**
     * CRF依存模型
     */
    public static String CRFDependencyModelPath = "hdfs://your/hadoop/abs/path/data/model/dependency/CRFDependencyModelMini.txt";
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

    public void resetConfig(){
        this.CoreDictionaryPath = this.HdfsRoot + "data/dictionary/CoreNatureDictionary.txt";
        this.CoreDictionaryTransformMatrixDictionaryPath = this.HdfsRoot + "data/dictionary/CoreNatureDictionary.tr.txt";
        this.CustomDictionaryPath[0] = this.HdfsRoot + "data/dictionary/custom/CustomDictionary.txt";
        this.BiGramDictionaryPath = this.HdfsRoot + "data/dictionary/CoreNatureDictionary.ngram.txt";
        this.CoreStopWordDictionaryPath = this.HdfsRoot + "data/dictionary/stopwords.txt";
        this.CoreSynonymDictionaryDictionaryPath = this.HdfsRoot + "data/dictionary/synonym/CoreSynonym.txt";
        this.PersonDictionaryPath = this.HdfsRoot + "data/dictionary/person/nr.txt";
        this.PersonDictionaryTrPath = this.HdfsRoot + "data/dictionary/person/nr.tr.txt";
        this.PlaceDictionaryPath = this.HdfsRoot + "data/dictionary/place/ns.txt";
        this.PlaceDictionaryTrPath = this.HdfsRoot + "data/dictionary/place/ns.tr.txt";
        this.OrganizationDictionaryPath = this.HdfsRoot + "data/dictionary/organization/nt.txt";
        this.OrganizationDictionaryTrPath = this.HdfsRoot + "data/dictionary/organization/nt.tr.txt";
        this.tcDictionaryRoot = this.HdfsRoot + "data/dictionary/tc/";
        this.SYTDictionaryPath = this.HdfsRoot + "data/dictionary/pinyin/SYTDictionary.txt";
        this.PinyinDictionaryPath = this.HdfsRoot + "data/dictionary/pinyin/pinyin.txt";
        this.TranslatedPersonDictionaryPath = this.HdfsRoot + "data/dictionary/person/nrf.txt";
        this.JapanesePersonDictionaryPath = this.HdfsRoot + "data/dictionary/person/nrj.txt";
        this.CharTypePath = this.HdfsRoot + "data/dictionary/other/CharType.bin";
        this.CharTablePath = this.HdfsRoot + "data/dictionary/other/CharTable.txt";
        this.WordNatureModelPath = this.HdfsRoot + "data/model/dependency/WordNature.txt";
        this.MaxEntModelPath = this.HdfsRoot + "data/model/dependency/MaxEntModel.txt";
        this.NNParserModelPath = this.HdfsRoot + "data/model/dependency/NNParserModel.txt";
        this.CRFSegmentModelPath = this.HdfsRoot + "data/model/segment/CRFSegmentModel.txt";
        this.HMMSegmentModelPath = this.HdfsRoot + "data/model/segment/HMMSegmentModel.bin";
        this.CRFDependencyModelPath = this.HdfsRoot + "data/model/dependency/CRFDependencyModelMini.txt";

    }

//    static
//    {
//        // 自动读取配置
//        Properties p = new Properties();
//        try
//        {
//            ClassLoader loader = Thread.currentThread().getContextClassLoader();
//            if (loader == null)
//            {  // IKVM (v.0.44.0.5) doesn't set context classloader
//                loader = Config.class.getClassLoader();
//            }
//            p.load(new InputStreamReader(Predefine.HANLP_PROPERTIES_PATH == null ?
//                loader.getResourceAsStream("hanlp.properties") :
//                new FileInputStream(Predefine.HANLP_PROPERTIES_PATH)
//                , "UTF-8"));
//            String root = p.getProperty("root", "").replaceAll("\\\\", "/");
//            if (root.length() > 0 && !root.endsWith("/")) root += "/";
//            CoreDictionaryPath = root + p.getProperty("CoreDictionaryPath", CoreDictionaryPath);
//            CoreDictionaryTransformMatrixDictionaryPath = root + p.getProperty("CoreDictionaryTransformMatrixDictionaryPath", CoreDictionaryTransformMatrixDictionaryPath);
//            BiGramDictionaryPath = root + p.getProperty("BiGramDictionaryPath", BiGramDictionaryPath);
//            CoreStopWordDictionaryPath = root + p.getProperty("CoreStopWordDictionaryPath", CoreStopWordDictionaryPath);
//            CoreSynonymDictionaryDictionaryPath = root + p.getProperty("CoreSynonymDictionaryDictionaryPath", CoreSynonymDictionaryDictionaryPath);
//            PersonDictionaryPath = root + p.getProperty("PersonDictionaryPath", PersonDictionaryPath);
//            PersonDictionaryTrPath = root + p.getProperty("PersonDictionaryTrPath", PersonDictionaryTrPath);
//            String[] pathArray = p.getProperty("CustomDictionaryPath", "data/dictionary/custom/CustomDictionary.txt").split(";");
//            String prePath = root;
//            for (int i = 0; i < pathArray.length; ++i)
//            {
//                if (pathArray[i].startsWith(" "))
//                {
//                    pathArray[i] = prePath + pathArray[i].trim();
//                }
//                else
//                {
//                    pathArray[i] = root + pathArray[i];
//                    int lastSplash = pathArray[i].lastIndexOf('/');
//                    if (lastSplash != -1)
//                    {
//                        prePath = pathArray[i].substring(0, lastSplash + 1);
//                    }
//                }
//            }
//            CustomDictionaryPath = pathArray;
//            tcDictionaryRoot = root + p.getProperty("tcDictionaryRoot", tcDictionaryRoot);
//            if (!tcDictionaryRoot.endsWith("/")) tcDictionaryRoot += '/';
//            SYTDictionaryPath = root + p.getProperty("SYTDictionaryPath", SYTDictionaryPath);
//            PinyinDictionaryPath = root + p.getProperty("PinyinDictionaryPath", PinyinDictionaryPath);
//            TranslatedPersonDictionaryPath = root + p.getProperty("TranslatedPersonDictionaryPath", TranslatedPersonDictionaryPath);
//            JapanesePersonDictionaryPath = root + p.getProperty("JapanesePersonDictionaryPath", JapanesePersonDictionaryPath);
//            PlaceDictionaryPath = root + p.getProperty("PlaceDictionaryPath", PlaceDictionaryPath);
//            PlaceDictionaryTrPath = root + p.getProperty("PlaceDictionaryTrPath", PlaceDictionaryTrPath);
//            OrganizationDictionaryPath = root + p.getProperty("OrganizationDictionaryPath", OrganizationDictionaryPath);
//            OrganizationDictionaryTrPath = root + p.getProperty("OrganizationDictionaryTrPath", OrganizationDictionaryTrPath);
//            CharTypePath = root + p.getProperty("CharTypePath", CharTypePath);
//            CharTablePath = root + p.getProperty("CharTablePath", CharTablePath);
//            WordNatureModelPath = root + p.getProperty("WordNatureModelPath", WordNatureModelPath);
//            MaxEntModelPath = root + p.getProperty("MaxEntModelPath", MaxEntModelPath);
//            NNParserModelPath = root + p.getProperty("NNParserModelPath", NNParserModelPath);
//            CRFSegmentModelPath = root + p.getProperty("CRFSegmentModelPath", CRFSegmentModelPath);
//            CRFDependencyModelPath = root + p.getProperty("CRFDependencyModelPath", CRFDependencyModelPath);
//            HMMSegmentModelPath = root + p.getProperty("HMMSegmentModelPath", HMMSegmentModelPath);
//            ShowTermNature = "true".equals(p.getProperty("ShowTermNature", "true"));
//            Normalization = "true".equals(p.getProperty("Normalization", "false"));
//            String ioAdapterClassName = p.getProperty("IOAdapter");
//            if (ioAdapterClassName != null)
//            {
//                try
//                {
//                    Class<?> clazz = Class.forName(ioAdapterClassName);
//                    Constructor<?> ctor = clazz.getConstructor();
//                    Object instance  = ctor.newInstance();
//                    if (instance != null) IOAdapter = (IIOAdapter) instance;
//                }
//                catch (ClassNotFoundException e)
//                {
//                    logger.warning(String.format("找不到IO适配器类： %s ，请检查第三方插件jar包", ioAdapterClassName));
//                }
//                catch (NoSuchMethodException e)
//                {
//                    logger.warning(String.format("工厂类[%s]没有默认构造方法，不符合要求", ioAdapterClassName));
//                }
//                catch (SecurityException e)
//                {
//                    logger.warning(String.format("工厂类[%s]默认构造方法无法访问，不符合要求", ioAdapterClassName));
//                }
//                catch (Exception e)
//                {
//                    logger.warning(String.format("工厂类[%s]构造失败：%s\n", ioAdapterClassName, TextUtility.exceptionToString(e)));
//                }
//            }
//        }
//        catch (Exception e)
//        {
//            StringBuilder sbInfo = new StringBuilder("========Tips========\n请将hanlp.properties放在下列目录：\n"); // 打印一些友好的tips
//            String classPath = (String) System.getProperties().get("java.class.path");
//            if (classPath != null)
//            {
//                for (String path : classPath.split(File.pathSeparator))
//                {
//                    if (new File(path).isDirectory())
//                    {
//                        sbInfo.append(path).append('\n');
//                    }
//                }
//            }
//            sbInfo.append("Web项目则请放到下列目录：\n" +
//                "Webapp/WEB-INF/lib\n" +
//                "Webapp/WEB-INF/classes\n" +
//                "Appserver/lib\n" +
//                "JRE/lib\n");
//            sbInfo.append("并且编辑root=PARENT/path/to/your/data\n");
//            sbInfo.append("现在HanLP将尝试从").append(System.getProperties().get("user.dir")).append("读取data……");
//            logger.severe("没有找到hanlp.properties，可能会导致找不到data\n" + sbInfo);
//        }
//    }

    /**
     * 开启调试模式(会降低性能)
     */
    public static void enableDebug()
    {
        enableDebug(true);
    }

    /**
     * 开启调试模式(会降低性能)
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
