package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.CWSEvaluator;

import java.io.IOException;

import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;


public class DemoEvaluateCWS
{
    public static void main(String[] args) throws IOException
    {
        logger.start("开始训练...\n");
        PerceptronTrainer trainer = new CWSTrainer();
        PerceptronTrainer.Result result = trainer.train(MSR.TRAIN_PATH, MSR.TRAIN_PATH, MSR.MODEL_PATH,
                                                        0.0, // 压缩比对准确率的影响很小
                                                        50, // 一般5个迭代就差不多收敛了
                                                        1 // 单线程的平均感知机算法收敛更稳定
        );
        logger.finish(" 训练完毕\n");

        Segment segment = new PerceptronLexicalAnalyzer(result.model).enableCustomDictionary(false); // 重要！必须禁用词典
        System.out.println(CWSEvaluator.evaluate(segment, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)); // 标准化评测
        // P:96.70 R:96.50 F1:96.60 OOV-R:70.80 IV-R:97.20
        // 受随机数影响，可能在96.60%左右波动
    }
}