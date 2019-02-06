/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dependency.perceptron.accessories.Edge;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;
import com.hankcs.hanlp.dependency.perceptron.structures.IndexMaps;
import com.hankcs.hanlp.dependency.perceptron.structures.ParserModel;
import com.hankcs.hanlp.dependency.perceptron.transition.features.FeatureExtractor;
import com.hankcs.hanlp.dependency.perceptron.accessories.CoNLLReader;
import com.hankcs.hanlp.dependency.perceptron.accessories.Pair;
import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.structures.Sentence;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.BeamElement;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Instance;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.State;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.concurrent.*;

public class KBeamArcEagerParser extends TransitionBasedParser
{
    ExecutorService executor;
    CompletionService<ArrayList<BeamElement>> pool;
    public Options options;

    public KBeamArcEagerParser(String modelPath) throws IOException, ClassNotFoundException
    {
        this(modelPath, Runtime.getRuntime().availableProcessors());
    }

    public KBeamArcEagerParser(String modelPath, int numOfThreads) throws IOException, ClassNotFoundException
    {
        this(new ParserModel(modelPath), numOfThreads);
    }

    public KBeamArcEagerParser(ParserModel parserModel, int numOfThreads)
    {
        this(new AveragedPerceptron(parserModel), parserModel.dependencyLabels, parserModel.shiftFeatureAveragedWeights.length, parserModel.maps, numOfThreads, parserModel.options);
    }

    public KBeamArcEagerParser(AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations,
                               int featureLength, IndexMaps maps, int numOfThreads, Options options)
    {
        super(classifier, dependencyRelations, featureLength, maps);
        executor = Executors.newFixedThreadPool(numOfThreads);
        pool = new ExecutorCompletionService<ArrayList<BeamElement>>(executor);
        this.options = options;
    }

    public Configuration parse(String[] words, String[] tags) throws ExecutionException, InterruptedException
    {
        return parse(maps.makeSentence(words, tags, options.rootFirst, options.lowercase), options.rootFirst, options.beamWidth, 1);
    }

    public Configuration parse(Sentence sentence) throws ExecutionException, InterruptedException
    {
        return parse(sentence, options.rootFirst, options.beamWidth, options.numOfThreads);
    }

    public Configuration parse(String[] words, String[] tags, boolean rootFirst, int beamWidth, int numOfThreads) throws ExecutionException, InterruptedException
    {
        return parse(maps.makeSentence(words, tags, options.rootFirst, options.lowercase), rootFirst, beamWidth, numOfThreads);
    }

    public Configuration parse(Sentence sentence, boolean rootFirst, int beamWidth, int numOfThreads) throws ExecutionException, InterruptedException
    {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);

        ArrayList<Configuration> beam = new ArrayList<Configuration>(beamWidth);
        beam.add(initialConfiguration);

        while (!ArcEager.isTerminal(beam))
        {
            TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();

            if (numOfThreads == 1)
            {
                sortBeam(beam, beamPreserver, false, new Instance(sentence, new HashMap<Integer, Edge>()), beamWidth, rootFirst, featureLength, classifier, dependencyRelations);
            }
            else
            {
                for (int b = 0; b < beam.size(); b++)
                {
                    pool.submit(new BeamScorerThread(true, classifier, beam.get(b),
                                                     dependencyRelations, featureLength, b, rootFirst));
                }
                fetchBeamFromPool(beamWidth, beam, beamPreserver);
            }


            beam = commitActionInBeam(beamWidth, beam, beamPreserver);
        }

        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam)
        {
            if (configuration.getScore(true) > bestScore)
            {
                bestScore = configuration.getScore(true);
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }

    private ArrayList<Configuration> commitActionInBeam(int beamWidth, ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver)
    {
        ArrayList<Configuration> repBeam = new ArrayList<Configuration>(beamWidth);
        for (BeamElement beamElement : beamPreserver.descendingSet())
        {
            if (repBeam.size() >= beamWidth)
                break;
            int b = beamElement.index;
            int action = beamElement.action;
            int label = beamElement.label;
            float score = beamElement.score;

            Configuration newConfig = beam.get(b).clone();

            ArcEager.commitAction(action, label, score, dependencyRelations, newConfig);
            repBeam.add(newConfig);
        }
        beam = repBeam;
        return beam;
    }

    private void parsePartialWithOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, Boolean isNonProjective, Instance instance, int beamWidth, boolean rootFirst)
    {
        sortBeam(beam, beamPreserver, isNonProjective, instance, beamWidth, rootFirst, featureLength, classifier, dependencyRelations);

        //todo
        if (beamPreserver.size() == 0)
        {
            ParseThread.sortBeam(beam, beamPreserver, false, null, beamWidth, rootFirst, featureLength, classifier, dependencyRelations);
        }
    }

    private static void sortBeam(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, Boolean isNonProjective, Instance instance, int beamWidth, boolean rootFirst, int featureLength, AveragedPerceptron classifier, Collection<Integer> dependencyRelations)
    {
        for (int b = 0; b < beam.size(); b++)
        {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Action.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Action.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Action.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Action.LeftArc, currentState);
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
            if (!canShift
                    && !canReduce
                    && !canRightArc
                    && !canLeftArc && rootFirst)
            {
                beamPreserver.add(new BeamElement(prevScore, b, 4, -1));

                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canShift)
            {
                if (isNonProjective || instance.actionCost(Action.Shift, -1, currentState) == 0)
                {
                    float score = classifier.shiftScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }

            if (canReduce)
            {
                if (isNonProjective || instance.actionCost(Action.Reduce, -1, currentState) == 0)
                {
                    float score = classifier.reduceScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }

            if (canRightArc)
            {
                float[] rightArcScores = classifier.rightArcScores(features, true);
                for (int dependency : dependencyRelations)
                {
                    if (isNonProjective || instance.actionCost(Action.RightArc, dependency, currentState) == 0)
                    {
                        float score = rightArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }

            if (canLeftArc)
            {
                float[] leftArcScores = classifier.leftArcScores(features, true);
                for (int dependency : dependencyRelations)
                {
                    if (isNonProjective || instance.actionCost(Action.LeftArc, dependency, currentState) == 0)
                    {
                        float score = leftArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
        }
    }

    public Configuration parsePartial(Instance instance, Sentence sentence, boolean rootFirst, int beamWidth, int numOfThreads) throws ExecutionException, InterruptedException
    {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        boolean isNonProjective = false;
        if (instance.isNonprojective())
        {
            isNonProjective = true;
        }

        ArrayList<Configuration> beam = new ArrayList<Configuration>(beamWidth);
        beam.add(initialConfiguration);

        while (!ArcEager.isTerminal(beam))
        {
            TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();

            if (numOfThreads == 1)
            {
                parsePartialWithOneThread(beam, beamPreserver, isNonProjective, instance, beamWidth, rootFirst);
            }
            else
            {
                for (int b = 0; b < beam.size(); b++)
                {
                    pool.submit(new PartialTreeBeamScorerThread(true, classifier, instance, beam.get(b),
                                                                dependencyRelations, featureLength, b));
                }
                fetchBeamFromPool(beamWidth, beam, beamPreserver);
            }

            beam = commitActionInBeam(beamWidth, beam, beamPreserver);
        }

        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam)
        {
            if (configuration.getScore(true) > bestScore)
            {
                bestScore = configuration.getScore(true);
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }

    private void fetchBeamFromPool(int beamWidth, ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver) throws InterruptedException, ExecutionException
    {
        for (int b = 0; b < beam.size(); b++)
        {
            for (BeamElement element : pool.take().get())
            {
                beamPreserver.add(element);
                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }
        }
    }

    public void parseConllFile(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean labeled, boolean lowerCased, int numThreads, boolean partial, String scorePath) throws IOException, ExecutionException, InterruptedException
    {
        if (numThreads == 1)
            parseConllFileNoParallel(inputFile, outputFile, rootFirst, beamWidth, labeled, lowerCased, numThreads, partial, scorePath);
        else
            parseConllFileParallel(inputFile, outputFile, rootFirst, beamWidth, lowerCased, numThreads, partial, scorePath);
    }

    /**
     * Needs Conll 2006 format
     *
     * @param inputFile
     * @param outputFile
     * @param rootFirst
     * @param beamWidth
     * @throws Exception
     */
    public void parseConllFileNoParallel(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean labeled, boolean lowerCased, int numOfThreads, boolean partial, String scorePath) throws IOException, ExecutionException, InterruptedException
    {
        CoNLLReader reader = new CoNLLReader(inputFile);
        boolean addScore = false;
        if (scorePath.trim().length() > 0)
            addScore = true;
        ArrayList<Float> scoreList = new ArrayList<Float>();

        long start = System.currentTimeMillis();
        int allArcs = 0;
        int size = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile + ".tmp"));
        int dataCount = 0;

        while (true)
        {
            ArrayList<Instance> data = reader.readData(15000, true, labeled, rootFirst, lowerCased, maps);
            size += data.size();
            if (data.size() == 0)
                break;

            for (Instance instance : data)
            {
                dataCount++;
                if (dataCount % 100 == 0)
                    System.err.print(dataCount + " ... ");
                Configuration bestParse;
                if (partial)
                    bestParse = parsePartial(instance, instance.getSentence(), rootFirst, beamWidth, numOfThreads);
                else bestParse = parse(instance.getSentence(), rootFirst, beamWidth, numOfThreads);

                int[] words = instance.getSentence().getWords();
                allArcs += words.length - 1;
                if (addScore)
                    scoreList.add(bestParse.score / bestParse.sentence.size());

                writeParsedSentence(writer, rootFirst, bestParse, words);
            }
        }

//        System.err.print("\n");
        long end = System.currentTimeMillis();
        float each = (1.0f * (end - start)) / size;
        float eacharc = (1.0f * (end - start)) / allArcs;

        writer.flush();
        writer.close();

//        DecimalFormat format = new DecimalFormat("##.00");
//
//        System.err.print(format.format(eacharc) + " ms for each arc!\n");
//        System.err.print(format.format(each) + " ms for each sentence!\n\n");

        BufferedReader gReader = new BufferedReader(new FileReader(inputFile));
        BufferedReader pReader = new BufferedReader(new FileReader(outputFile + ".tmp"));
        BufferedWriter pwriter = new BufferedWriter(new FileWriter(outputFile));

        String line;

        while ((line = pReader.readLine()) != null)
        {
            String gLine = gReader.readLine();
            if (line.trim().length() > 0)
            {
                while (gLine.trim().length() == 0)
                    gLine = gReader.readLine();
                String[] ps = line.split("\t");
                String[] gs = gLine.split("\t");
                gs[6] = ps[0];
                gs[7] = ps[1];
                StringBuilder output = new StringBuilder();
                for (int i = 0; i < gs.length; i++)
                {
                    output.append(gs[i]).append("\t");
                }
                pwriter.write(output.toString().trim() + "\n");
            }
            else
            {
                pwriter.write("\n");
            }
        }
        pwriter.flush();
        pwriter.close();

        if (addScore)
        {
            BufferedWriter scoreWriter = new BufferedWriter(new FileWriter(scorePath));

            for (int i = 0; i < scoreList.size(); i++)
                scoreWriter.write(scoreList.get(i) + "\n");
            scoreWriter.flush();
            scoreWriter.close();
        }
        IOUtil.deleteFile(outputFile + ".tmp");
    }

    private void writeParsedSentence(BufferedWriter writer, boolean rootFirst, Configuration bestParse, int[] words) throws IOException
    {
        StringBuilder finalOutput = new StringBuilder();
        for (int i = 0; i < words.length; i++)
        {
            int w = i + 1;
            int head = bestParse.state.getHead(w);
            int dep = bestParse.state.getDependent(w);

            if (w == bestParse.state.rootIndex && !rootFirst)
                continue;

            if (head == bestParse.state.rootIndex)
                head = 0;

            String label = head == 0 ? maps.rootString : maps.idWord[dep];
            String output = head + "\t" + label + "\n";
            finalOutput.append(output);
        }
        finalOutput.append("\n");
        writer.write(finalOutput.toString());
    }

    public void parseTaggedFile(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean lowerCased, String separator, int numOfThreads) throws Exception
    {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        long start = System.currentTimeMillis();

        ExecutorService executor = Executors.newFixedThreadPool(numOfThreads);
        CompletionService<Pair<String, Integer>> pool = new ExecutorCompletionService<Pair<String, Integer>>(executor);


        String line;
        int count = 0;
        int lineNum = 0;
        while ((line = reader.readLine()) != null)
        {
            pool.submit(new ParseTaggedThread(lineNum++, line, separator, rootFirst, lowerCased, maps, beamWidth, this));

            if (lineNum % 1000 == 0)
            {
                String[] outs = new String[lineNum];
                for (int i = 0; i < lineNum; i++)
                {
                    count++;
                    if (count % 100 == 0)
                        System.err.print(count + "...");
                    Pair<String, Integer> result = pool.take().get();
                    outs[result.second] = result.first;
                }

                for (int i = 0; i < lineNum; i++)
                {
                    if (outs[i].length() > 0)
                    {
                        writer.write(outs[i]);
                    }
                }

                lineNum = 0;
            }
        }

        if (lineNum > 0)
        {
            String[] outs = new String[lineNum];
            for (int i = 0; i < lineNum; i++)
            {
                count++;
                if (count % 100 == 0)
                    System.err.print(count + "...");
                Pair<String, Integer> result = pool.take().get();
                outs[result.second] = result.first;
            }

            for (int i = 0; i < lineNum; i++)
            {

                if (outs[i].length() > 0)
                {
                    writer.write(outs[i]);
                }
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("\n" + (end - start) + " ms");
        writer.flush();
        writer.close();

        System.out.println("done!");
    }

    public void parseConllFileParallel(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean lowerCased, int numThreads, boolean partial, String scorePath) throws IOException, InterruptedException, ExecutionException
    {
        CoNLLReader reader = new CoNLLReader(inputFile);

        boolean addScore = false;
        if (scorePath.trim().length() > 0)
            addScore = true;
        ArrayList<Float> scoreList = new ArrayList<Float>();

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Pair<Configuration, Integer>> pool = new ExecutorCompletionService<Pair<Configuration, Integer>>(executor);

        long start = System.currentTimeMillis();
        int allArcs = 0;
        int size = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile + ".tmp"));
        int dataCount = 0;

        while (true)
        {
            ArrayList<Instance> data = reader.readData(15000, true, true, rootFirst, lowerCased, maps);
            size += data.size();
            if (data.size() == 0)
                break;

            int index = 0;
            Configuration[] confs = new Configuration[data.size()];

            for (Instance instance : data)
            {
                ParseThread thread = new ParseThread(index, classifier, dependencyRelations, featureLength, instance.getSentence(), rootFirst, beamWidth, instance, partial);
                pool.submit(thread);
                index++;
            }

            for (int i = 0; i < confs.length; i++)
            {
                dataCount++;
//                if (dataCount % 100 == 0)
//                    System.err.print(dataCount + " ... ");

                Pair<Configuration, Integer> configurationIntegerPair = pool.take().get();
                confs[configurationIntegerPair.second] = configurationIntegerPair.first;
            }

            for (int j = 0; j < confs.length; j++)
            {
                Configuration bestParse = confs[j];
                if (addScore)
                {
                    scoreList.add(bestParse.score / bestParse.sentence.size());
                }
                int[] words = data.get(j).getSentence().getWords();

                allArcs += words.length - 1;

                writeParsedSentence(writer, rootFirst, bestParse, words);
            }
        }

//        System.err.print("\n");
        long end = System.currentTimeMillis();
        float each = (1.0f * (end - start)) / size;
        float eacharc = (1.0f * (end - start)) / allArcs;

        writer.flush();
        writer.close();

//        DecimalFormat format = new DecimalFormat("##.00");
//
//        System.err.print(format.format(eacharc) + " ms for each arc!\n");
//        System.err.print(format.format(each) + " ms for each sentence!\n\n");

        BufferedReader gReader = new BufferedReader(new FileReader(inputFile));
        BufferedReader pReader = new BufferedReader(new FileReader(outputFile + ".tmp"));
        BufferedWriter pwriter = new BufferedWriter(new FileWriter(outputFile));

        String line;

        while ((line = pReader.readLine()) != null)
        {
            String gLine = gReader.readLine();
            if (line.trim().length() > 0)
            {
                while (gLine.trim().length() == 0)
                    gLine = gReader.readLine();
                String[] ps = line.split("\t");
                String[] gs = gLine.split("\t");
                gs[6] = ps[0];
                gs[7] = ps[1];
                StringBuilder output = new StringBuilder();
                for (int i = 0; i < gs.length; i++)
                {
                    output.append(gs[i]).append("\t");
                }
                pwriter.write(output.toString().trim() + "\n");
            }
            else
            {
                pwriter.write("\n");
            }
        }
        pwriter.flush();
        pwriter.close();

        if (addScore)
        {
            BufferedWriter scoreWriter = new BufferedWriter(new FileWriter(scorePath));

            for (int i = 0; i < scoreList.size(); i++)
                scoreWriter.write(scoreList.get(i) + "\n");
            scoreWriter.flush();
            scoreWriter.close();
        }
        IOUtil.deleteFile(outputFile + ".tmp");
    }

    public void shutDownLiveThreads()
    {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated)
        {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }
}