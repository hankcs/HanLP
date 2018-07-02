package com.hankcs.hanlp.model.crf.crfpp;

import com.hankcs.hanlp.corpus.io.IOUtil;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * 训练入口
 *
 * @author zhifac
 */
public class Encoder
{
    public static int MODEL_VERSION = 100;

    public enum Algorithm
    {
        CRF_L2, CRF_L1, MIRA;

        public static Algorithm fromString(String algorithm)
        {
            algorithm = algorithm.toLowerCase();
            if (algorithm.equals("crf") || algorithm.equals("crf-l2"))
            {
                return Encoder.Algorithm.CRF_L2;
            }
            else if (algorithm.equals("crf-l1"))
            {
                return Encoder.Algorithm.CRF_L1;
            }
            else if (algorithm.equals("mira"))
            {
                return Encoder.Algorithm.MIRA;
            }
            throw new IllegalArgumentException("invalid algorithm: " + algorithm);
        }
    }

    public Encoder()
    {
    }

    /**
     * 训练
     *
     * @param templFile     模板文件
     * @param trainFile     训练文件
     * @param modelFile     模型文件
     * @param textModelFile 是否输出文本形式的模型文件
     * @param maxitr        最大迭代次数
     * @param freq          特征最低频次
     * @param eta           收敛阈值
     * @param C             cost-factor
     * @param threadNum     线程数
     * @param shrinkingSize
     * @param algorithm     训练算法
     * @return
     */
    public boolean learn(String templFile, String trainFile, String modelFile, boolean textModelFile,
                         int maxitr, int freq, double eta, double C, int threadNum, int shrinkingSize,
                         Algorithm algorithm)
    {
        if (eta <= 0)
        {
            System.err.println("eta must be > 0.0");
            return false;
        }
        if (C < 0.0)
        {
            System.err.println("C must be >= 0.0");
            return false;
        }
        if (shrinkingSize < 1)
        {
            System.err.println("shrinkingSize must be >= 1");
            return false;
        }
        if (threadNum <= 0)
        {
            System.err.println("thread must be  > 0");
            return false;
        }
        EncoderFeatureIndex featureIndex = new EncoderFeatureIndex(threadNum);
        List<TaggerImpl> x = new ArrayList<TaggerImpl>();
        if (!featureIndex.open(templFile, trainFile))
        {
            System.err.println("Fail to open " + templFile + " " + trainFile);
        }
//        File file = new File(trainFile);
//        if (!file.exists())
//        {
//            System.err.println("train file " + trainFile + " does not exist.");
//            return false;
//        }
        BufferedReader br = null;
        try
        {
            InputStreamReader isr = new InputStreamReader(IOUtil.newInputStream(trainFile), "UTF-8");
            br = new BufferedReader(isr);
            int lineNo = 0;
            while (true)
            {
                TaggerImpl tagger = new TaggerImpl(TaggerImpl.Mode.LEARN);
                tagger.open(featureIndex);
                TaggerImpl.ReadStatus status = tagger.read(br);
                if (status == TaggerImpl.ReadStatus.ERROR)
                {
                    System.err.println("error when reading " + trainFile);
                    return false;
                }
                if (!tagger.empty())
                {
                    if (!tagger.shrink())
                    {
                        System.err.println("fail to build feature index ");
                        return false;
                    }
                    tagger.setThread_id_(lineNo % threadNum);
                    x.add(tagger);
                }
                else if (status == TaggerImpl.ReadStatus.EOF)
                {
                    break;
                }
                else
                {
                    continue;
                }
                if (++lineNo % 100 == 0)
                {
                    System.out.print(lineNo + ".. ");
                }
            }
            br.close();
        }
        catch (IOException e)
        {
            System.err.println("train file " + trainFile + " does not exist.");
            return false;
        }
        featureIndex.shrink(freq, x);

        double[] alpha = new double[featureIndex.size()];
        Arrays.fill(alpha, 0.0);
        featureIndex.setAlpha_(alpha);

        System.out.println("Number of sentences: " + x.size());
        System.out.println("Number of features:  " + featureIndex.size());
        System.out.println("Number of thread(s): " + threadNum);
        System.out.println("Freq:                " + freq);
        System.out.println("eta:                 " + eta);
        System.out.println("C:                   " + C);
        System.out.println("shrinking size:      " + shrinkingSize);

        switch (algorithm)
        {
            case CRF_L1:
                if (!runCRF(x, featureIndex, alpha, maxitr, C, eta, shrinkingSize, threadNum, true))
                {
                    System.err.println("CRF_L1 execute error");
                    return false;
                }
                break;
            case CRF_L2:
                if (!runCRF(x, featureIndex, alpha, maxitr, C, eta, shrinkingSize, threadNum, false))
                {
                    System.err.println("CRF_L2 execute error");
                    return false;
                }
                break;
            case MIRA:
                if (!runMIRA(x, featureIndex, alpha, maxitr, C, eta, shrinkingSize, threadNum))
                {
                    System.err.println("MIRA execute error");
                    return false;
                }
                break;
            default:
                break;
        }

        if (!featureIndex.save(modelFile, textModelFile))
        {
            System.err.println("Failed to save model");
        }
        System.out.println("Done!");
        return true;
    }

    /**
     * CRF训练
     *
     * @param x             句子列表
     * @param featureIndex  特征编号表
     * @param alpha         特征函数的代价
     * @param maxItr        最大迭代次数
     * @param C             cost factor
     * @param eta           收敛阈值
     * @param shrinkingSize 未使用
     * @param threadNum     线程数
     * @param orthant       是否使用L1范数
     * @return 是否成功
     */
    private boolean runCRF(List<TaggerImpl> x,
                           EncoderFeatureIndex featureIndex,
                           double[] alpha,
                           int maxItr,
                           double C,
                           double eta,
                           int shrinkingSize,
                           int threadNum,
                           boolean orthant)
    {
        double oldObj = 1e+37;
        int converge = 0;
        LbfgsOptimizer lbfgs = new LbfgsOptimizer();
        List<CRFEncoderThread> threads = new ArrayList<CRFEncoderThread>();

        for (int i = 0; i < threadNum; i++)
        {
            CRFEncoderThread thread = new CRFEncoderThread(alpha.length);
            thread.start_i = i;
            thread.size = x.size();
            thread.threadNum = threadNum;
            thread.x = x;
            threads.add(thread);
        }

        int all = 0;
        for (int i = 0; i < x.size(); i++)
        {
            all += x.get(i).size();
        }

        ExecutorService executor = Executors.newFixedThreadPool(threadNum);
        for (int itr = 0; itr < maxItr; itr++)
        {
            featureIndex.clear();

            try
            {
                executor.invokeAll(threads);
            }
            catch (Exception e)
            {
                e.printStackTrace();
                return false;
            }

            for (int i = 1; i < threadNum; i++)
            {
                threads.get(0).obj += threads.get(i).obj;
                threads.get(0).err += threads.get(i).err;
                threads.get(0).zeroone += threads.get(i).zeroone;
            }
            for (int i = 1; i < threadNum; i++)
            {
                for (int k = 0; k < featureIndex.size(); k++)
                {
                    threads.get(0).expected[k] += threads.get(i).expected[k];
                }
            }
            int numNonZero = 0;
            if (orthant)
            {
                for (int k = 0; k < featureIndex.size(); k++)
                {
                    threads.get(0).obj += Math.abs(alpha[k] / C);
                    if (alpha[k] != 0.0)
                    {
                        numNonZero++;
                    }
                }
            }
            else
            {
                numNonZero = featureIndex.size();
                for (int k = 0; k < featureIndex.size(); k++)
                {
                    threads.get(0).obj += (alpha[k] * alpha[k] / (2.0 * C));
                    threads.get(0).expected[k] += alpha[k] / C;
                }
            }
            for (int i = 1; i < threadNum; i++)
            {
                // try to free some memory
                threads.get(i).expected = null;
            }

            double diff = (itr == 0 ? 1.0 : Math.abs(oldObj - threads.get(0).obj) / oldObj);
            StringBuilder b = new StringBuilder();
            b.append("iter=").append(itr);
            b.append(" terr=").append(1.0 * threads.get(0).err / all);
            b.append(" serr=").append(1.0 * threads.get(0).zeroone / x.size());
            b.append(" act=").append(numNonZero);
            b.append(" obj=").append(threads.get(0).obj);
            b.append(" diff=").append(diff);
            System.out.println(b.toString());

            oldObj = threads.get(0).obj;

            if (diff < eta)
            {
                converge++;
            }
            else
            {
                converge = 0;
            }

            if (itr > maxItr || converge == 3)
            {
                break;
            }

            int ret = lbfgs.optimize(featureIndex.size(), alpha, threads.get(0).obj, threads.get(0).expected, orthant, C);
            if (ret <= 0)
            {
                return false;
            }
        }
        executor.shutdown();
        try
        {
            executor.awaitTermination(-1, TimeUnit.SECONDS);
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.err.println("fail waiting executor to shutdown");
        }
        return true;
    }

    public boolean runMIRA(List<TaggerImpl> x,
                           EncoderFeatureIndex featureIndex,
                           double[] alpha,
                           int maxItr,
                           double C,
                           double eta,
                           int shrinkingSize,
                           int threadNum)
    {
        Integer[] shrinkArr = new Integer[x.size()];
        Arrays.fill(shrinkArr, 0);
        List<Integer> shrink = Arrays.asList(shrinkArr);
        Double[] upperArr = new Double[x.size()];
        Arrays.fill(upperArr, 0.0);
        List<Double> upperBound = Arrays.asList(upperArr);
        Double[] expectArr = new Double[featureIndex.size()];
        List<Double> expected = Arrays.asList(expectArr);

        if (threadNum > 1)
        {
            System.err.println("WARN: MIRA does not support multi-threading");
        }
        int converge = 0;
        int all = 0;
        for (int i = 0; i < x.size(); i++)
        {
            all += x.get(i).size();
        }

        for (int itr = 0; itr < maxItr; itr++)
        {
            int zeroone = 0;
            int err = 0;
            int activeSet = 0;
            int upperActiveSet = 0;
            double maxKktViolation = 0.0;

            for (int i = 0; i < x.size(); i++)
            {
                if (shrink.get(i) >= shrinkingSize)
                {
                    continue;
                }
                ++activeSet;
                for (int t = 0; t < expected.size(); t++)
                {
                    expected.set(t, 0.0);
                }
                double costDiff = x.get(i).collins(expected);
                int errorNum = x.get(i).eval();
                err += errorNum;
                if (errorNum != 0)
                {
                    ++zeroone;
                }
                if (errorNum == 0)
                {
                    shrink.set(i, shrink.get(i) + 1);
                }
                else
                {
                    shrink.set(i, 0);
                    double s = 0.0;
                    for (int k = 0; k < expected.size(); k++)
                    {
                        s += expected.get(k) * expected.get(k);
                    }
                    double mu = Math.max(0.0, (errorNum - costDiff) / s);

                    if (upperBound.get(i) + mu > C)
                    {
                        mu = C - upperBound.get(i);
                        upperActiveSet++;
                    }
                    else
                    {
                        maxKktViolation = Math.max(errorNum - costDiff, maxKktViolation);
                    }

                    if (mu > 1e-10)
                    {
                        upperBound.set(i, upperBound.get(i) + mu);
                        upperBound.set(i, Math.min(C, upperBound.get(i)));
                        for (int k = 0; k < expected.size(); k++)
                        {
                            alpha[k] += mu * expected.get(k);
                        }
                    }
                }
            }
            double obj = 0.0;
            for (int i = 0; i < featureIndex.size(); i++)
            {
                obj += alpha[i] * alpha[i];
            }

            StringBuilder b = new StringBuilder();
            b.append("iter=").append(itr);
            b.append(" terr=").append(1.0 * err / all);
            b.append(" serr=").append(1.0 * zeroone / x.size());
            b.append(" act=").append(activeSet);
            b.append(" uact=").append(upperActiveSet);
            b.append(" obj=").append(obj);
            b.append(" kkt=").append(maxKktViolation);
            System.out.println(b.toString());

            if (maxKktViolation <= 0.0)
            {
                for (int i = 0; i < shrink.size(); i++)
                {
                    shrink.set(i, 0);
                }
                converge++;
            }
            else
            {
                converge = 0;
            }
            if (itr > maxItr || converge == 2)
            {
                break;
            }
        }
        return true;
    }

    public static void main(String[] args)
    {
        if (args.length < 3)
        {
            System.err.println("incorrect No. of args");
            return;
        }
        String templFile = args[0];
        String trainFile = args[1];
        String modelFile = args[2];
        Encoder enc = new Encoder();
        long time1 = new Date().getTime();
        if (!enc.learn(templFile, trainFile, modelFile, false, 100000, 1, 0.0001, 1.0, 1, 20, Algorithm.CRF_L2))
        {
            System.err.println("error training model");
            return;
        }
        System.out.println(new Date().getTime() - time1);
    }
}
