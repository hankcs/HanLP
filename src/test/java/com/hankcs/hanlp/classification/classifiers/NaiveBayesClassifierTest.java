package com.hankcs.hanlp.classification.classifiers;

import com.hankcs.hanlp.classification.models.NaiveBayesModel;
import com.hankcs.hanlp.classification.utilities.TextProcessUtility;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.util.Map;

import static com.hankcs.demo.DemoTextClassification.CORPUS_FOLDER;


public class NaiveBayesClassifierTest extends TestCase
{
    private static final String MODEL_PATH = "data/test/classification.ser";
    private Map<String, String[]> trainingDataSet;


    private void loadDataSet()
    {
        if (trainingDataSet != null) return;
        System.out.printf("正在从 %s 中加载分类语料...\n", CORPUS_FOLDER);
        trainingDataSet = TextProcessUtility.loadCorpus(CORPUS_FOLDER);
        for (Map.Entry<String, String[]> entry : trainingDataSet.entrySet())
        {
            System.out.printf("%s : %d 个文档\n", entry.getKey(), entry.getValue().length);
        }
    }

    public void testTrain() throws Exception
    {
        loadDataSet();
        NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier();
        long start = System.currentTimeMillis();
        System.out.println("开始训练...");
        naiveBayesClassifier.train(trainingDataSet);
        System.out.printf("训练耗时：%d ms\n", System.currentTimeMillis() - start);
        // 将模型保存
        IOUtil.saveObjectTo(naiveBayesClassifier.getNaiveBayesModel(), MODEL_PATH);
    }

    public void testPredictAndAccuracy() throws Exception
    {
        // 加载模型
        NaiveBayesModel model = (NaiveBayesModel) IOUtil.readObjectFrom(MODEL_PATH);
        if (model == null)
        {
            testTrain();
            model = (NaiveBayesModel) IOUtil.readObjectFrom(MODEL_PATH);
        }
        NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier(model);
        // 预测单个文档
        String path = CORPUS_FOLDER + "/财经/12.txt";
        String text = IOUtil.readTxt(path);
        String label = naiveBayesClassifier.classify(text);
        String title = text.split("\\n")[0].replaceAll("\\s", "");
        System.out.printf("《%s》 属于分类 【%s】\n", title, label);
        text = "2016年中国铁路完成固定资产投资将达8000亿元";
        title = text;
        label = naiveBayesClassifier.classify(text);
        System.out.printf("《%s》 属于分类 【%s】\n", title, label);
        text = "国安2016赛季年票开售比赛场次减少套票却上涨";
        title = text;
        label = naiveBayesClassifier.classify(text);
        System.out.printf("《%s》 属于分类 【%s】\n", title, label);
        // 对将训练集作为测试，计算准确率
        int totalDocuments = 0;
        int rightDocuments = 0;
        loadDataSet();
        long start = System.currentTimeMillis();
        System.out.println("开始评测...");
        for (Map.Entry<String, String[]> entry : trainingDataSet.entrySet())
        {
            String category = entry.getKey();
            String[] documents = entry.getValue();

            totalDocuments += documents.length;
            for (String document : documents)
            {
                if (category.equals(naiveBayesClassifier.classify(document))) ++rightDocuments;
            }
        }
        System.out.printf("准确率 %d / %d = %.2f%%\n速度 %.2f 文档/秒", rightDocuments, totalDocuments,
                          rightDocuments / (double) totalDocuments * 100.,
                          totalDocuments / (double) (System.currentTimeMillis() - start) * 1000.
        );
    }

    public void testPredict() throws Exception
    {
        // 加载模型
        NaiveBayesModel model = (NaiveBayesModel) IOUtil.readObjectFrom(MODEL_PATH);
        if (model == null)
        {
            testTrain();
            model = (NaiveBayesModel) IOUtil.readObjectFrom(MODEL_PATH);
        }
        NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier(model);
        Map<String, Double> pMap = naiveBayesClassifier.predict("国安2016赛季年票开售比赛场次减少套票却上涨");
        for (Map.Entry<String, Double> entry : pMap.entrySet())
        {
            System.out.println(entry);
        }
    }
}