package com.hankcs.hanlp.model.perceptron;

import java.io.IOException;

public class DemoTrainCWS
{
    public static void main(String[] args) throws IOException
    {
        PerceptronTrainer trainer = new CWSTrainer();
        PerceptronTrainer.Result result = trainer.train(
                "data/test/pku98/199801.txt",
                Config.CWS_MODEL_FILE
        );
        System.out.printf("准确率F1:%.2f\n", result.getAccuracy());
        PerceptronSegmenter segment = new PerceptronSegmenter(result.getModel());
        // 也可以用
//        Segment segment = new AveragedPerceptronSegment(POS_MODEL_FILE);
        System.out.println(segment.segment("商品与服务"));
    }
}