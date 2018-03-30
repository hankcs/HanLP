package com.hankcs.hanlp.model.crf.crfpp;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * @author zhifac
 */
public class CRFEncoderThread implements Callable<Integer>
{
    public List<TaggerImpl> x;
    public int start_i;
    public int wSize;
    public int threadNum;
    public int zeroone;
    public int err;
    public int size;
    public double obj;
    public double[] expected;

    public CRFEncoderThread(int wsize)
    {
        if (wsize > 0)
        {
            this.wSize = wsize;
            expected = new double[wsize];
            Arrays.fill(expected, 0.0);
        }
    }

    public Integer call()
    {
        obj = 0.0;
        err = zeroone = 0;
        if (expected == null)
        {
            expected = new double[wSize];
        }
        Arrays.fill(expected, 0.0);
        for (int i = start_i; i < size; i = i + threadNum)
        {
            obj += x.get(i).gradient(expected);
            int errorNum = x.get(i).eval();
            x.get(i).clearNodes();
            err += errorNum;
            if (errorNum != 0)
            {
                ++zeroone;
            }
        }
        return err;
    }
}
