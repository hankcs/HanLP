/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.hankcs.hanlp.dependency.perceptron.transition.trainer;

import com.hankcs.hanlp.classification.utilities.io.ConsoleLogger;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dependency.perceptron.accessories.Edge;
import com.hankcs.hanlp.dependency.perceptron.accessories.Evaluator;
import com.hankcs.hanlp.dependency.perceptron.accessories.Pair;
import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;
import com.hankcs.hanlp.dependency.perceptron.structures.IndexMaps;
import com.hankcs.hanlp.dependency.perceptron.structures.ParserModel;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.BeamElement;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Instance;
import com.hankcs.hanlp.dependency.perceptron.transition.features.FeatureExtractor;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.*;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;
import com.hankcs.hanlp.dependency.perceptron.transition.configuration.State;
import com.hankcs.hanlp.model.perceptron.feature.FeatureSortItem;
import com.hankcs.hanlp.utility.MathUtility;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;

import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;

public class ArcEagerBeamTrainer extends TransitionBasedParser
{
    Options options;
    /**
     * Can be either "early" or "max_violation"
     * For more information read:
     * Liang Huang, Suphan Fayong and Yang Guo. "Structured perceptron with inexact search."
     * In Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,
     * pp. 142-151. Association for Computational Linguistics, 2012.
     */
    private String updateMode;
    private Random randGen;

    public ArcEagerBeamTrainer(String updateMode, AveragedPerceptron classifier, Options options,
                               ArrayList<Integer> dependencyRelations, int featureLength, IndexMaps maps)
    {
        super(classifier, dependencyRelations, featureLength, maps);
        this.updateMode = updateMode;
        this.options = options;
        randGen = new Random();
    }

    public void train(ArrayList<Instance> trainData, String devPath, int maxIteration, String modelPath, boolean lowerCased, HashSet<String> punctuations, int partialTreeIter) throws IOException, ExecutionException, InterruptedException
    {
        /**
         * Actions: 0=shift, 1=reduce, 2=unshift, ra_dep=3+dep, la_dep=3+dependencyRelations.size()+dep
         */
        ExecutorService executor = Executors.newFixedThreadPool(options.numOfThreads);
        CompletionService<ArrayList<BeamElement>> pool = new ExecutorCompletionService<ArrayList<BeamElement>>(executor);

        double bestUAS = -1.;
        for (int i = 1; i <= maxIteration; i++)
        {
            long start = System.currentTimeMillis();

            int dataCount = 0;

            int logEvery = (int) Math.ceil(trainData.size() / 10000f);
            for (Instance instance : trainData)
            {
                dataCount++;
                if (dataCount % logEvery == 0 || dataCount == trainData.size())
                {
                    System.out.printf("\r迭代 " + i + "/" + maxIteration + " %.2f%% ", MathUtility.percentage(dataCount, trainData.size()));
                }
                trainOnOneSample(instance, partialTreeIter, i, dataCount, pool);

                classifier.incrementIteration();
            }
//            System.out.print("\n");
            long end = System.currentTimeMillis();
            long timeSec = (end - start) / 1000;
            System.out.print(" 耗时 " + timeSec + " 秒。");

//            System.out.print("saving the model...");
            ParserModel parserModel = new ParserModel(classifier, maps, dependencyRelations, options);
//            infStruct.saveModel(modelPath + "_iter" + i);

//            System.out.println("done\n");

            if (!devPath.equals(""))
            {
                AveragedPerceptron averagedPerceptron = new AveragedPerceptron(parserModel);

//                int raSize = averagedPerceptron.raSize();
//                int effectiveRaSize = averagedPerceptron.effectiveRaSize();
//                float raRatio = 100.0f * effectiveRaSize / raSize;
//
//                int laSize = averagedPerceptron.laSize();
//                int effectiveLaSize = averagedPerceptron.effectiveLaSize();
//                float laRatio = 100.0f * effectiveLaSize / laSize;

//                DecimalFormat format = new DecimalFormat("##.00");
//                System.out.println("size of RA features in memory:" + effectiveRaSize + "/" + raSize + "=" + format.format(raRatio) + "%");
//                System.out.println("size of LA features in memory:" + effectiveLaSize + "/" + laSize + "=" + format.format(laRatio) + "%");
                KBeamArcEagerParser parser = new KBeamArcEagerParser(averagedPerceptron, dependencyRelations, featureLength, maps, options.numOfThreads, options);

                String outputFile = modelPath + ".__tmp__";
                parser.parseConllFile(devPath, outputFile,
                                      options.rootFirst, options.beamWidth, true, lowerCased, options.numOfThreads, false, "");
                double[] score = Evaluator.evaluate(devPath, outputFile, punctuations);
                System.out.printf("UAS=%.2f LAS=%.2f", score[0], score[1]);
                IOUtil.deleteFile(outputFile);
                parser.shutDownLiveThreads();
                if (score[0] > bestUAS)
                {
                    bestUAS = score[0];
                    System.out.println(" 最高分！保存中...");
                    parserModel.saveModel(modelPath);
                }
                else
                {
                    System.out.println();
                }
            }
            else
            {
                parserModel.saveModel(modelPath);
                System.out.println();
            }
        }
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated)
        {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    /**
     * 在线学习
     *
     * @param instance        实例
     * @param partialTreeIter 半标注树的训练迭代数
     * @param i               当前迭代数
     * @param dataCount
     * @param pool
     * @throws Exception
     */
    private void trainOnOneSample(Instance instance, int partialTreeIter, int i, int dataCount, CompletionService<ArrayList<BeamElement>> pool) throws InterruptedException, ExecutionException
    {
        boolean isPartial = instance.isPartial(options.rootFirst);

        if (partialTreeIter > i && isPartial)
            return;

        Configuration initialConfiguration = new Configuration(instance.getSentence(), options.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<Configuration>(options.beamWidth);
        beam.add(initialConfiguration);

        /**
         * The float is the oracle's cost
         * For more information see:
         * Yoav Goldberg and Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic Oracles."
         * TACL 1 (2013): 403-414.
         * for the mean while we just use zero-cost oracles
         */
        Collection<Configuration> oracles = new HashSet<Configuration>();

        oracles.add(firstOracle);

        /**
         * For keeping track of the violations
         * For more information see:
         * Liang Huang, Suphan Fayong and Yang Guo. "Structured perceptron with inexact search."
         * In Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,
         * pp. 142-151. Association for Computational Linguistics, 2012.
         */
        float maxViol = Float.NEGATIVE_INFINITY;
        Pair<Configuration, Configuration> maxViolPair = null;

        Configuration bestScoringOracle = null;
        boolean oracleInBeam = false;

        while (!ArcEager.isTerminal(beam) && beam.size() > 0)
        {
            /**
             *  generating new oracles
             *  it keeps the oracles which are in the terminal state
             */
            Collection<Configuration> newOracles = new HashSet<Configuration>();

            if (options.useDynamicOracle || isPartial)
            {
                bestScoringOracle = zeroCostDynamicOracle(instance, oracles, newOracles);
            }
            else
            {
                bestScoringOracle = staticOracle(instance, oracles, newOracles);
            }
            // try explore non-optimal transitions

            if (newOracles.size() == 0)
            {
//                System.err.println("...no oracle(" + dataCount + ")...");
                bestScoringOracle = staticOracle(instance, oracles, newOracles);
            }
            oracles = newOracles;

            TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();

            if (options.numOfThreads == 1 || beam.size() == 1)
            {
                beamSortOneThread(beam, beamPreserver);
            }
            else
            {
                for (int b = 0; b < beam.size(); b++)
                {
                    pool.submit(new BeamScorerThread(false, classifier, beam.get(b),
                                                     dependencyRelations, featureLength, b, options.rootFirst));
                }
                for (int b = 0; b < beam.size(); b++)
                {
                    for (BeamElement element : pool.take().get())
                    {
                        beamPreserver.add(element);
                        if (beamPreserver.size() > options.beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }

            if (beamPreserver.size() == 0 || beam.size() == 0)
            {
                break;
            }
            else
            {
                oracleInBeam = false;

                ArrayList<Configuration> repBeam = new ArrayList<Configuration>(options.beamWidth);
                for (BeamElement beamElement : beamPreserver.descendingSet())
                {
//                    if (repBeam.size() >= options.beamWidth) // 只要beamWidth个configuration（这句是多余的）
//                        break;
                    int b = beamElement.index;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    float score = beamElement.score;

                    Configuration newConfig = beam.get(b).clone();

                    ArcEager.commitAction(action, label, score, dependencyRelations, newConfig);
                    repBeam.add(newConfig);

                    if (!oracleInBeam && oracles.contains(newConfig))
                        oracleInBeam = true;
                }
                beam = repBeam;

                if (beam.size() > 0 && oracles.size() > 0)
                {
                    Configuration bestConfig = beam.get(0);
                    if (oracles.contains(bestConfig)) // 模型认为的最大分值configuration是zero cost
                    {
                        oracles = new HashSet<Configuration>();
                        oracles.add(bestConfig);
                    }
                    else // 否则
                    {
                        if (options.useRandomOracleSelection) // 随机选择一个 oracle
                        { // choosing randomly, otherwise using latent structured Perceptron
                            List<Configuration> keys = new ArrayList<Configuration>(oracles);
                            Configuration randomKey = keys.get(randGen.nextInt(keys.size()));
                            oracles = new HashSet<Configuration>();
                            oracles.add(randomKey);
                            bestScoringOracle = randomKey;
                        }
                        else // 选择 oracle中被模型认为分值最大的那个
                        {
                            oracles = new HashSet<Configuration>();
                            oracles.add(bestScoringOracle);
                        }
                    }

                    // do early update
                    if (!oracleInBeam && updateMode.equals("early"))
                        break;

                    // keep violations
                    if (!oracleInBeam && updateMode.equals("max_violation"))
                    {
                        float violation = bestConfig.getScore(true) - bestScoringOracle.getScore(true);//Math.abs(beam.get(0).getScore(true) - bestScoringOracle.getScore(true));
                        if (violation > maxViol)
                        {
                            maxViol = violation;
                            maxViolPair = new Pair<Configuration, Configuration>(bestConfig, bestScoringOracle);
                        }
                    }
                }
                else
                    break;
            }
        }

        // updating weights
        if (!oracleInBeam ||
                !bestScoringOracle.equals(beam.get(0)) // 虽然oracle在beam里面，但在最后时刻，它的得分不是最高
                )
        {
            updateWeights(initialConfiguration, maxViol, isPartial, bestScoringOracle, maxViolPair, beam);
        }
    }

    private Configuration staticOracle(Instance instance, Collection<Configuration> oracles, Collection<Configuration> newOracles)
    {
        Configuration bestScoringOracle = null;
        int top = -1;
        int first = -1;
        HashMap<Integer, Edge> goldDependencies = instance.getGoldDependencies();
        HashMap<Integer, HashSet<Integer>> reversedDependencies = instance.getReversedDependencies();

        for (Configuration configuration : oracles)
        {
            State state = configuration.state;
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

            if (!state.stackEmpty())
                top = state.stackTop();
            if (!state.bufferEmpty())
                first = state.bufferHead();

            if (!configuration.state.isTerminalState())
            {
                Configuration newConfig = configuration.clone();

                if (first > 0 && goldDependencies.containsKey(first) && goldDependencies.get(first).headIndex == top)
                {
                    int dependency = goldDependencies.get(first).relationId;
                    float[] scores = classifier.rightArcScores(features, false);
                    float score = scores[dependency];
                    ArcEager.rightArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependency);
                    newConfig.addScore(score);
                }
                else if (top > 0 && goldDependencies.containsKey(top) && goldDependencies.get(top).headIndex == first)
                {
                    int dependency = goldDependencies.get(top).relationId;
                    float[] scores = classifier.leftArcScores(features, false);
                    float score = scores[dependency];
                    ArcEager.leftArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependencyRelations.size() + dependency);
                    newConfig.addScore(score);
                }
                else if (top >= 0 && state.hasHead(top))
                {

                    if (reversedDependencies.containsKey(top))
                    {
                        if (reversedDependencies.get(top).size() == state.valence(top))
                        {
                            float score = classifier.reduceScore(features, false);
                            ArcEager.reduce(newConfig.state);
                            newConfig.addAction(1);
                            newConfig.addScore(score);
                        }
                        else
                        {
                            float score = classifier.shiftScore(features, false);
                            ArcEager.shift(newConfig.state);
                            newConfig.addAction(0);
                            newConfig.addScore(score);
                        }
                    }
                    else
                    {
                        float score = classifier.reduceScore(features, false);
                        ArcEager.reduce(newConfig.state);
                        newConfig.addAction(1);
                        newConfig.addScore(score);
                    }

                }
                else if (state.bufferEmpty() && state.stackSize() == 1 && state.stackTop() == state.rootIndex)
                {
                    float score = classifier.reduceScore(features, false);
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                }
                else
                {
                    float score = classifier.shiftScore(features, true);
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                }
                bestScoringOracle = newConfig;
                newOracles.add(newConfig);
            }
            else
            {
                newOracles.add(configuration);
            }
        }
        return bestScoringOracle;
    }

    /**
     * 获取 zero cost oracle
     *
     * @param instance   训练实例
     * @param oracles    当前的oracle
     * @param newOracles 储存新oracle
     * @return 这些 oracles 中在模型看来分数最大的那个
     * @throws Exception
     */
    private Configuration zeroCostDynamicOracle(Instance instance, Collection<Configuration> oracles, Collection<Configuration> newOracles)
    {
        float bestScore = Float.NEGATIVE_INFINITY;
        Configuration bestScoringOracle = null;

        for (Configuration configuration : oracles)
        {
            if (!configuration.state.isTerminalState())
            {
                State currentState = configuration.state;
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                // I only assumed that we need zero cost ones
                if (instance.actionCost(Action.Shift, -1, currentState) == 0)
                {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.shiftScore(features, false);
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                    newOracles.add(newConfig);

                    if (newConfig.getScore(true) > bestScore)
                    {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                }
                if (ArcEager.canDo(Action.RightArc, currentState))
                {
                    float[] rightArcScores = classifier.rightArcScores(features, false);
                    for (int dependency : dependencyRelations)
                    {
                        if (instance.actionCost(Action.RightArc, dependency, currentState) == 0)
                        {
                            Configuration newConfig = configuration.clone();
                            float score = rightArcScores[dependency];
                            ArcEager.rightArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependency);
                            newConfig.addScore(score);
                            newOracles.add(newConfig);

                            if (newConfig.getScore(true) > bestScore)
                            {
                                bestScore = newConfig.getScore(true);
                                bestScoringOracle = newConfig;
                            }
                        }
                    }
                }
                if (ArcEager.canDo(Action.LeftArc, currentState))
                {
                    float[] leftArcScores = classifier.leftArcScores(features, false);

                    for (int dependency : dependencyRelations)
                    {
                        if (instance.actionCost(Action.LeftArc, dependency, currentState) == 0)
                        {
                            Configuration newConfig = configuration.clone();
                            float score = leftArcScores[dependency];
                            ArcEager.leftArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependencyRelations.size() + dependency);
                            newConfig.addScore(score);
                            newOracles.add(newConfig);

                            if (newConfig.getScore(true) > bestScore)
                            {
                                bestScore = newConfig.getScore(true);
                                bestScoringOracle = newConfig;
                            }
                        }
                    }
                }
                if (instance.actionCost(Action.Reduce, -1, currentState) == 0)
                {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.reduceScore(features, false);
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                    newOracles.add(newConfig);

                    if (newConfig.getScore(true) > bestScore)
                    {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                }
            }
            else
            {
                newOracles.add(configuration);
            }
        }

        return bestScoringOracle;
    }

    /**
     * 每个beam元素执行所有可能的动作一次
     *
     * @param beam
     * @param beamPreserver
     * @throws Exception
     */
    private void beamSortOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver)
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

            if (canShift)
            {
                float score = classifier.shiftScore(features, false);
                float addedScore = score + prevScore;
                addToBeam(beamPreserver, b, addedScore, 0, -1, options.beamWidth);
            }
            if (canReduce)
            {
                float score = classifier.reduceScore(features, false);
                float addedScore = score + prevScore;
                addToBeam(beamPreserver, b, addedScore, 1, -1, options.beamWidth);
            }

            if (canRightArc)
            {
                float[] rightArcScores = classifier.rightArcScores(features, false);
                for (int dependency : dependencyRelations)
                {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    addToBeam(beamPreserver, b, addedScore, 2, dependency, options.beamWidth);
                }
            }
            if (canLeftArc)
            {
                float[] leftArcScores = classifier.leftArcScores(features, false);
                for (int dependency : dependencyRelations)
                {
                    float score = leftArcScores[dependency];
                    float addedScore = score + prevScore;
                    addToBeam(beamPreserver, b, addedScore, 3, dependency, options.beamWidth);
                }
            }
        }
    }

    private void addToBeam(TreeSet<BeamElement> beamPreserver, int b, float addedScore, int action, int label, int beamWidth)
    {
        beamPreserver.add(new BeamElement(addedScore, b, action, label));

        if (beamPreserver.size() > beamWidth)
            beamPreserver.pollFirst();
    }

    private void updateWeights(Configuration initialConfiguration, float maxViol, boolean isPartial, Configuration bestScoringOracle, Pair<Configuration, Configuration> maxViolPair, ArrayList<Configuration> beam)
    {
        Configuration predicted;
        Configuration finalOracle;
        if (!updateMode.equals("max_violation"))
        {
            finalOracle = bestScoringOracle;
            predicted = beam.get(0);
        }
        else
        {
            float violation = beam.get(0).getScore(true) - bestScoringOracle.getScore(true); //Math.abs(beam.get(0).getScore(true) - bestScoringOracle.getScore(true));
            if (violation > maxViol)
            {
                maxViolPair = new Pair<Configuration, Configuration>(beam.get(0), bestScoringOracle);
            }
            predicted = maxViolPair.first;
            finalOracle = maxViolPair.second;
        }

        Object[] predictedFeatures = new Object[featureLength];
        Object[] oracleFeatures = new Object[featureLength];
        for (int f = 0; f < predictedFeatures.length; f++)
        {
            oracleFeatures[f] = new HashMap<Pair<Integer, Long>, Float>();
            predictedFeatures[f] = new HashMap<Pair<Integer, Long>, Float>();
        }

        Configuration predictedConfiguration = initialConfiguration.clone();
        Configuration oracleConfiguration = initialConfiguration.clone();

        for (int action : finalOracle.actionHistory)
        {
            boolean isTrueFeature = isTrueFeature(isPartial, oracleConfiguration, action);

            if (isTrueFeature)
            {   // if the made dependency is truly for the word
                Object[] feats = FeatureExtractor.extractAllParseFeatures(oracleConfiguration, featureLength);
                for (int f = 0; f < feats.length; f++)
                {
                    Pair<Integer, Object> featName = new Pair<Integer, Object>(action, feats[f]);
                    HashMap<Pair<Integer, Object>, Float> map = (HashMap<Pair<Integer, Object>, Float>) oracleFeatures[f];
                    Float value = map.get(featName);
                    if (value == null)
                        map.put(featName, 1.0f);
                    else
                        map.put(featName, value + 1);
                }
            }

            if (action == 0)
            {
                ArcEager.shift(oracleConfiguration.state);
            }
            else if (action == 1)
            {
                ArcEager.reduce(oracleConfiguration.state);
            }
            else if (action >= (3 + dependencyRelations.size()))
            {
                int dependency = action - (3 + dependencyRelations.size());
                ArcEager.leftArc(oracleConfiguration.state, dependency);
            }
            else if (action >= 3)
            {
                int dependency = action - 3;
                ArcEager.rightArc(oracleConfiguration.state, dependency);
            }
        }

        for (int action : predicted.actionHistory)
        {
            boolean isTrueFeature = isTrueFeature(isPartial, predictedConfiguration, action);

            if (isTrueFeature)
            {   // if the made dependency is truely for the word
                Object[] feats = FeatureExtractor.extractAllParseFeatures(predictedConfiguration, featureLength);
                if (action != 2) // do not take into account for unshift
                    for (int f = 0; f < feats.length; f++)
                    {
                        Pair<Integer, Object> featName = new Pair<Integer, Object>(action, feats[f]);
                        HashMap<Pair<Integer, Object>, Float> map = (HashMap<Pair<Integer, Object>, Float>) predictedFeatures[f];
                        Float value = map.get(featName);
                        if (value == null)
                            map.put(featName, 1.f);
                        else
                            map.put(featName, map.get(featName) + 1);
                    }
            }

            State state = predictedConfiguration.state;
            if (action == 0)
            {
                ArcEager.shift(state);
            }
            else if (action == 1)
            {
                ArcEager.reduce(state);
            }
            else if (action >= 3 + dependencyRelations.size())
            {
                int dependency = action - (3 + dependencyRelations.size());
                ArcEager.leftArc(state, dependency);
            }
            else if (action >= 3)
            {
                int dependency = action - 3;
                ArcEager.rightArc(state, dependency);
            }
            else if (action == 2)
            {
                ArcEager.unShift(state);
            }
        }

        for (int f = 0; f < predictedFeatures.length; f++)
        {
            HashMap<Pair<Integer, Object>, Float> map = (HashMap<Pair<Integer, Object>, Float>) predictedFeatures[f];
            HashMap<Pair<Integer, Object>, Float> map2 = (HashMap<Pair<Integer, Object>, Float>) oracleFeatures[f];
            for (Pair<Integer, Object> feat : map.keySet())
            {
                int action = feat.first;
                LabeledAction labeledAction = new LabeledAction(action, dependencyRelations.size());
                Action actionType = labeledAction.action;
                int dependency = labeledAction.label;

                if (feat.second != null)
                {
                    Object feature = feat.second;
                    if (!(map2.containsKey(feat) && map2.get(feat).equals(map.get(feat))))
                        classifier.changeWeight(actionType, f, feature, dependency, -map.get(feat));
                }
            }

            for (Pair<Integer, Object> feat : map2.keySet())
            {
                int action = feat.first;
                LabeledAction labeledAction = new LabeledAction(action, dependencyRelations.size());
                Action actionType = labeledAction.action;
                int dependency = labeledAction.label;

                if (feat.second != null)
                {
                    Object feature = feat.second;
                    if (!(map.containsKey(feat) && map.get(feat).equals(map2.get(feat))))
                        classifier.changeWeight(actionType, f, feature, dependency, map2.get(feat));
                }
            }
        }
    }

    private static boolean isTrueFeature(boolean isPartial, Configuration oracleConfiguration, int action)
    {
        boolean isTrueFeature = true;
        if (isPartial && action >= 3)
        {
            if (!oracleConfiguration.state.hasHead(oracleConfiguration.state.stackTop()) || !oracleConfiguration.state.hasHead(oracleConfiguration.state.bufferHead()))
                isTrueFeature = false;
        }
        else if (isPartial && action == 0)
        {
            if (!oracleConfiguration.state.hasHead(oracleConfiguration.state.bufferHead()))
                isTrueFeature = false;
        }
        else if (isPartial && action == 1)
        {
            if (!oracleConfiguration.state.hasHead(oracleConfiguration.state.stackTop()))
                isTrueFeature = false;
        }
        return isTrueFeature;
    }

}