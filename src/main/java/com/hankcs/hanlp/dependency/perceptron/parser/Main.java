/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.parser;

import com.hankcs.hanlp.dependency.perceptron.structures.IndexMaps;
import com.hankcs.hanlp.dependency.perceptron.accessories.CoNLLReader;
import com.hankcs.hanlp.dependency.perceptron.accessories.Evaluator;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;
import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.structures.ParserModel;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Instance;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.KBeamArcEagerParser;
import com.hankcs.hanlp.dependency.perceptron.transition.trainer.ArcEagerBeamTrainer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;

public class Main
{
    public static void main(String[] args) throws Exception
    {
        Options options = Options.processArgs(args);

        if (options.showHelp)
        {
            Options.showHelp();
        }
        else
        {
            System.out.println(options);
            if (options.train)
            {
                train(options);
            }
            else if (options.parseTaggedFile || options.parseConllFile || options.parsePartialConll)
            {
                parse(options);
            }
            else if (options.evaluate)
            {
                evaluate(options);
            }
            else
            {
                Options.showHelp();
            }
        }
        System.exit(0);
    }

    private static void evaluate(Options options) throws Exception
    {
        if (options.goldFile.equals("") || options.predFile.equals(""))
            Options.showHelp();
        else
        {
            Evaluator.evaluate(options.goldFile, options.predFile, options.punctuations);
        }
    }

    private static void parse(Options options) throws Exception
    {
        if (options.outputFile.equals("") || options.inputFile.equals("")
                || options.modelFile.equals(""))
        {
            Options.showHelp();

        }
        else
        {
            ParserModel parserModel = new ParserModel(options.modelFile);
            ArrayList<Integer> dependencyLabels = parserModel.dependencyLabels;
            IndexMaps maps = parserModel.maps;


            Options inf_options = parserModel.options;
            AveragedPerceptron averagedPerceptron = new AveragedPerceptron(parserModel);

            int featureSize = averagedPerceptron.featureSize();
            KBeamArcEagerParser parser = new KBeamArcEagerParser(averagedPerceptron, dependencyLabels, featureSize, maps, options.numOfThreads, options);

            if (options.parseTaggedFile)
                parser.parseTaggedFile(options.inputFile,
                                       options.outputFile, inf_options.rootFirst, inf_options.beamWidth, inf_options.lowercase, options.separator, options.numOfThreads);
            else if (options.parseConllFile)
                parser.parseConllFile(options.inputFile,
                                      options.outputFile, inf_options.rootFirst, inf_options.beamWidth, true, inf_options.lowercase, options.numOfThreads, false, options.scorePath);
            else if (options.parsePartialConll)
                parser.parseConllFile(options.inputFile,
                                      options.outputFile, inf_options.rootFirst, inf_options.beamWidth, options.labeled, inf_options.lowercase, options.numOfThreads, true, options.scorePath);
            parser.shutDownLiveThreads();
        }
    }

    public static void train(Options options) throws IOException, ExecutionException, InterruptedException
    {
        if (options.inputFile.equals("") || options.modelFile.equals(""))
        {
            Options.showHelp();
        }
        else
        {
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, options.clusterFile);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<Instance> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options.rootFirst, options.lowercase, maps);
//            System.out.println("读取CoNLL文件结束。");

            ArrayList<Integer> dependencyLabels = new ArrayList<Integer>();
            dependencyLabels.addAll(maps.getLabels().keySet());

            int featureLength = options.useExtendedFeatures ? 72 : 26;
            if (options.useExtendedWithBrownClusterFeatures || maps.hasClusters())
                featureLength = 153;

            System.out.println("训练集句子数量: " + dataSet.size());

            HashMap<String, Integer> labels = new HashMap<String, Integer>();
            labels.put("sh", labels.size());
            labels.put("rd", labels.size());
            labels.put("us", labels.size());
            for (int label : dependencyLabels)
            {
                if (options.labeled)
                {
                    labels.put("ra_" + label, 3 + label);
                    labels.put("la_" + label, 3 + dependencyLabels.size() + label);
                }
                else
                {
                    labels.put("ra_" + label, 3);
                    labels.put("la_" + label, 4);
                }
            }

            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                                                                  new AveragedPerceptron(featureLength, dependencyLabels.size()),
                                                                  options, dependencyLabels, featureLength, maps);
            trainer.train(dataSet, options.devPath, options.trainingIter, options.modelFile, options.lowercase, options.punctuations, options.partialTrainingStartingIteration);
        }
    }
}
