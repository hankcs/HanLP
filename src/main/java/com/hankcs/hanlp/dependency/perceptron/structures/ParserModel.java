package com.hankcs.hanlp.dependency.perceptron.structures;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;
import com.hankcs.hanlp.dependency.perceptron.learning.AveragedPerceptron;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 1/8/15
 * Time: 11:41 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

/**
 * 句法分析模型（参数、超参数、词表等）
 */
public class ParserModel
{
    public HashMap<Object, Float>[] shiftFeatureAveragedWeights;
    public HashMap<Object, Float>[] reduceFeatureAveragedWeights;
    public HashMap<Object, CompactArray>[] leftArcFeatureAveragedWeights;
    public HashMap<Object, CompactArray>[] rightArcFeatureAveragedWeights;
    public int dependencySize;

    public IndexMaps maps;
    public ArrayList<Integer> dependencyLabels;
    public Options options;

    public ParserModel(HashMap<Object, Float>[] shiftFeatureAveragedWeights, HashMap<Object, Float>[] reduceFeatureAveragedWeights, HashMap<Object, CompactArray>[] leftArcFeatureAveragedWeights, HashMap<Object, CompactArray>[] rightArcFeatureAveragedWeights,
                       IndexMaps maps, ArrayList<Integer> dependencyLabels, Options options, int dependencySize)
    {
        this.shiftFeatureAveragedWeights = shiftFeatureAveragedWeights;
        this.reduceFeatureAveragedWeights = reduceFeatureAveragedWeights;
        this.leftArcFeatureAveragedWeights = leftArcFeatureAveragedWeights;
        this.rightArcFeatureAveragedWeights = rightArcFeatureAveragedWeights;
        this.maps = maps;
        this.dependencyLabels = dependencyLabels;
        this.options = options;
        this.dependencySize = dependencySize;
    }

    public ParserModel(AveragedPerceptron perceptron, IndexMaps maps, ArrayList<Integer> dependencyLabels, Options options)
    {
        shiftFeatureAveragedWeights = new HashMap[perceptron.shiftFeatureAveragedWeights.length];
        reduceFeatureAveragedWeights = new HashMap[perceptron.reduceFeatureAveragedWeights.length];

        HashMap<Object, Float>[] map = perceptron.shiftFeatureWeights;
        HashMap<Object, Float>[] avgMap = perceptron.shiftFeatureAveragedWeights;
        this.dependencySize = perceptron.dependencySize;

        for (int i = 0; i < shiftFeatureAveragedWeights.length; i++)
        {
            shiftFeatureAveragedWeights[i] = new HashMap<Object, Float>();
            for (Object feat : map[i].keySet())
            {
                float vals = map[i].get(feat);
                float avgVals = avgMap[i].get(feat);
                float newVals = vals - (avgVals / perceptron.iteration);
                shiftFeatureAveragedWeights[i].put(feat, newVals);
            }
        }

        HashMap<Object, Float>[] map4 = perceptron.reduceFeatureWeights;
        HashMap<Object, Float>[] avgMap4 = perceptron.reduceFeatureAveragedWeights;
        this.dependencySize = perceptron.dependencySize;

        for (int i = 0; i < reduceFeatureAveragedWeights.length; i++)
        {
            reduceFeatureAveragedWeights[i] = new HashMap<Object, Float>();
            for (Object feat : map4[i].keySet())
            {
                float vals = map4[i].get(feat);
                float avgVals = avgMap4[i].get(feat);
                float newVals = vals - (avgVals / perceptron.iteration);
                reduceFeatureAveragedWeights[i].put(feat, newVals);
            }
        }

        leftArcFeatureAveragedWeights = new HashMap[perceptron.leftArcFeatureAveragedWeights.length];
        HashMap<Object, CompactArray>[] map2 = perceptron.leftArcFeatureWeights;
        HashMap<Object, CompactArray>[] avgMap2 = perceptron.leftArcFeatureAveragedWeights;

        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++)
        {
            leftArcFeatureAveragedWeights[i] = new HashMap<Object, CompactArray>();
            for (Object feat : map2[i].keySet())
            {
                CompactArray vals = map2[i].get(feat);
                CompactArray avgVals = avgMap2[i].get(feat);
                leftArcFeatureAveragedWeights[i].put(feat, getAveragedCompactArray(vals, avgVals, perceptron.iteration));
            }
        }

        rightArcFeatureAveragedWeights = new HashMap[perceptron.rightArcFeatureAveragedWeights.length];
        HashMap<Object, CompactArray>[] map3 = perceptron.rightArcFeatureWeights;
        HashMap<Object, CompactArray>[] avgMap3 = perceptron.rightArcFeatureAveragedWeights;

        for (int i = 0; i < rightArcFeatureAveragedWeights.length; i++)
        {
            rightArcFeatureAveragedWeights[i] = new HashMap<Object, CompactArray>();
            for (Object feat : map3[i].keySet())
            {
                CompactArray vals = map3[i].get(feat);
                CompactArray avgVals = avgMap3[i].get(feat);
                rightArcFeatureAveragedWeights[i].put(feat, getAveragedCompactArray(vals, avgVals, perceptron.iteration));
            }
        }

        this.maps = maps;
        this.dependencyLabels = dependencyLabels;
        this.options = options;
    }

    public ParserModel(String modelPath) throws IOException, ClassNotFoundException
    {
        ObjectInputStream reader = new ObjectInputStream(new GZIPInputStream(IOUtil.newInputStream(modelPath)));
        dependencyLabels = (ArrayList<Integer>) reader.readObject();
        maps = (IndexMaps) reader.readObject();
        options = (Options) reader.readObject();
        shiftFeatureAveragedWeights = (HashMap<Object, Float>[]) reader.readObject();
        reduceFeatureAveragedWeights = (HashMap<Object, Float>[]) reader.readObject();
        leftArcFeatureAveragedWeights = (HashMap<Object, CompactArray>[]) reader.readObject();
        rightArcFeatureAveragedWeights = (HashMap<Object, CompactArray>[]) reader.readObject();
        dependencySize = reader.readInt();
        reader.close();
    }

    public void saveModel(String modelPath) throws IOException
    {
        ObjectOutput writer = new ObjectOutputStream(new GZIPOutputStream(IOUtil.newOutputStream(modelPath)));
        writer.writeObject(dependencyLabels);
        writer.writeObject(maps);
        writer.writeObject(options);
        writer.writeObject(shiftFeatureAveragedWeights);
        writer.writeObject(reduceFeatureAveragedWeights);
        writer.writeObject(leftArcFeatureAveragedWeights);
        writer.writeObject(rightArcFeatureAveragedWeights);
        writer.writeInt(dependencySize);
        writer.close();
    }

    private CompactArray getAveragedCompactArray(CompactArray ca, CompactArray aca, int iteration)
    {
        int offset = ca.getOffset();
        float[] a = ca.getArray();
        float[] aa = aca.getArray();
        float[] aNew = new float[a.length];
        for (int i = 0; i < a.length; i++)
        {
            aNew[i] = a[i] - (aa[i] / iteration);
        }
        return new CompactArray(offset, aNew);
    }
}
