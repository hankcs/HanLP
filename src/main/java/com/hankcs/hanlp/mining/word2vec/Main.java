package com.hankcs.hanlp.mining.word2vec;

import java.io.File;

public class Main
{
    public static void main(String argv[]) 
    {
        if (argv.length != 2)
        {
            System.err.printf("usage:\t%s [train_file] [model_path]\n" +
                                      "\t\t[train_file] is the path to corpus.\n" +
                                      "\t\t[model_path] is where we save model.\n",
                              Main.class.getName()
            );
            return;
        }
        String trainFile = argv[0];
        if (!new File(trainFile).exists())
        {
            System.err.printf("corpus %s does not exist.\n", trainFile);
            return;
        }
        String modelFile = argv[1];
        File folder = new File(modelFile).getParentFile();
        if (folder == null) folder = new File("./");
        if (!folder.exists())
        {
            if (!folder.mkdirs())
            {
                System.err.printf("failed to create folder %s\n", folder.getAbsolutePath());
                return;
            }
        }
        Word2VecTrainer builder = new Word2VecTrainer();
        builder.train(trainFile, modelFile);
    }
}
