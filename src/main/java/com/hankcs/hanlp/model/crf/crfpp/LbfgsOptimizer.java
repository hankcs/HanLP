package com.hankcs.hanlp.model.crf.crfpp;

import java.util.Arrays;

/**
 * @author zhifac
 */
public class LbfgsOptimizer
{
    int iflag_, iscn, nfev, iycn, point, npt, iter, info, ispt, isyt, iypt, maxfev;
    double stp, stp1;
    double[] diag_ = null;
    double[] w_ = null;
    double[] v_ = null;
    double[] xi_ = null;
    Mcsrch mcsrch_ = null;

    public void pseudo_gradient(int size,
                                double[] v,
                                double[] x,
                                double[] g,
                                double C)
    {
        for (int i = 0; i < size; ++i)
        {
            if (x[i] == 0)
            {
                if (g[i] + C < 0)
                {
                    v[i] = g[i] + C;
                }
                else if (g[i] - C > 0)
                {
                    v[i] = g[i] - C;
                }
                else
                {
                    v[i] = 0;
                }
            }
            else
            {
                v[i] = g[i] + C * Mcsrch.sigma(x[i]);
            }
        }
    }

    int lbfgs_optimize(int size,
                       int msize,
                       double[] x,
                       double f,
                       double[] g,
                       double[] diag,
                       double[] w, boolean orthant, double C,
                       double[] v, double[] xi, int iflag)
    {
        double yy = 0.0;
        double ys = 0.0;
        int bound = 0;
        int cp = 0;

        if (orthant)
        {
            pseudo_gradient(size, v, x, g, C);
        }

        if (mcsrch_ == null)
        {
            mcsrch_ = new Mcsrch();
        }

        boolean firstLoop = true;

        // initialization
        if (iflag == 0)
        {
            point = 0;
            for (int i = 0; i < size; ++i)
            {
                diag[i] = 1.0;
            }
            ispt = size + (msize << 1);
            iypt = ispt + size * msize;
            for (int i = 0; i < size; ++i)
            {
                w[ispt + i] = -v[i] * diag[i];
            }
            stp1 = 1.0 / Math.sqrt(Mcsrch.ddot_(size, v, 0, v, 0));
        }

        // MAIN ITERATION LOOP
        while (true)
        {
            if (!firstLoop || (firstLoop && iflag != 1 && iflag != 2))
            {
                ++iter;
                info = 0;
                if (orthant)
                {
                    for (int i = 0; i < size; ++i)
                    {
                        xi[i] = (x[i] != 0 ? Mcsrch.sigma(x[i]) : Mcsrch.sigma(-v[i]));
                    }
                }
                if (iter != 1)
                {
                    if (iter > size) bound = size;

                    // COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
                    // "Updating quasi-Newton matrices with limited storage",
                    // Mathematics of Computation, Vol.24, No.151, pp. 773-782.
                    ys = Mcsrch.ddot_(size, w, iypt + npt, w, ispt + npt);
                    yy = Mcsrch.ddot_(size, w, iypt + npt, w, iypt + npt);
                    for (int i = 0; i < size; ++i)
                    {
                        diag[i] = ys / yy;
                    }
                }
            }
            if (iter != 1 && (!firstLoop || (iflag != 1 && firstLoop)))
            {
                cp = point;
                if (point == 0)
                {
                    cp = msize;
                }
                w[size + cp - 1] = 1.0 / ys;

                for (int i = 0; i < size; ++i)
                {
                    w[i] = -v[i];
                }

                bound = Math.min(iter - 1, msize);

                cp = point;
                for (int i = 0; i < bound; ++i)
                {
                    --cp;
                    if (cp == -1)
                    {
                        cp = msize - 1;
                    }
                    double sq = Mcsrch.ddot_(size, w, ispt + cp * size, w, 0);
                    int inmc = size + msize + cp;
                    iycn = iypt + cp * size;
                    w[inmc] = w[size + cp] * sq;
                    double d = -w[inmc];
                    Mcsrch.daxpy_(size, d, w, iycn, w, 0);
                }

                for (int i = 0; i < size; ++i)
                {
                    w[i] = diag[i] * w[i];
                }

                for (int i = 0; i < bound; ++i)
                {
                    double yr = Mcsrch.ddot_(size, w, iypt + cp * size, w, 0);
                    double beta = w[size + cp] * yr;
                    int inmc = size + msize + cp;
                    beta = w[inmc] - beta;
                    iscn = ispt + cp * size;
                    Mcsrch.daxpy_(size, beta, w, iscn, w, 0);
                    ++cp;
                    if (cp == msize)
                    {
                        cp = 0;
                    }
                }

                if (orthant)
                {
                    for (int i = 0; i < size; ++i)
                    {
                        w[i] = (Mcsrch.sigma(w[i]) == Mcsrch.sigma(-v[i]) ? w[i] : 0);
                    }
                }
                // STORE THE NEW SEARCH DIRECTION
                for (int i = 0; i < size; ++i)
                {
                    w[ispt + point * size + i] = w[i];
                }
            }
            // OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION
            // BY USING THE LINE SEARCH ROUTINE MCSRCH
            if (!firstLoop || (firstLoop && iflag != 1))
            {
                nfev = 0;
                stp = 1.0;
                if (iter == 1)
                {
                    stp = stp1;
                }
                for (int i = 0; i < size; ++i)
                {
                    w[i] = g[i];
                }
            }
            double[] stpArr = {stp};
            int[] infoArr = {info};
            int[] nfevArr = {nfev};

            mcsrch_.mcsrch(size, x, f, v, w, ispt + point * size,
                           stpArr, infoArr, nfevArr, diag);
            stp = stpArr[0];
            info = infoArr[0];
            nfev = nfevArr[0];

            if (info == -1)
            {
                if (orthant)
                {
                    for (int i = 0; i < size; ++i)
                    {
                        x[i] = (Mcsrch.sigma(x[i]) == Mcsrch.sigma(xi[i]) ? x[i] : 0);
                    }
                }
                return 1; // next value
            }
            if (info != 1)
            {
                System.err.println("The line search routine mcsrch failed: error code:" + info);
                return -1;
            }

            // COMPUTE THE NEW STEP AND GRADIENT CHANGE
            npt = point * size;
            for (int i = 0; i < size; ++i)
            {
                w[ispt + npt + i] = stp * w[ispt + npt + i];
                w[iypt + npt + i] = g[i] - w[i];
            }
            ++point;
            if (point == msize) point = 0;

            double gnorm = Math.sqrt(Mcsrch.ddot_(size, v, 0, v, 0));
            double xnorm = Math.max(1.0, Math.sqrt(Mcsrch.ddot_(size, x, 0, x, 0)));
            if (gnorm / xnorm <= Mcsrch.eps)
            {
                return 0; // OK terminated
            }

            firstLoop = false;
        }
    }


    public LbfgsOptimizer()
    {
        iflag_ = iscn = nfev = 0;
        iycn = point = npt = iter = info = ispt = isyt = iypt = maxfev = 0;
        mcsrch_ = null;
    }

    public void clear()
    {
        iflag_ = iscn = nfev = iycn = point = npt =
            iter = info = ispt = isyt = iypt = 0;
        stp = stp1 = 0.0;
        diag_ = null;
        w_ = null;
        v_ = null;
        mcsrch_ = null;
    }

    public int init(int n, int m)
    {
        //This is old interface for backword compatibility
        final int msize = 5;
        final int size = n;
        iflag_ = 0;
        w_ = new double[size * (2 * msize + 1) + 2 * msize];
        Arrays.fill(w_, 0.0);
        diag_ = new double[size];
        v_ = new double[size];
        return 0;
    }

    public int optimize(double[] x, double f, double[] g)
    {
        return optimize(diag_.length, x, f, g, false, 1.0);
    }

    public int optimize(int size, double[] x, double f, double[] g, boolean orthant, double C)
    {
        int msize = 5;
        if (w_ == null)
        {
            iflag_ = 0;
            w_ = new double[size * (2 * msize + 1) + 2 * msize];
            Arrays.fill(w_, 0.0);
            diag_ = new double[size];
            v_ = new double[size];
            if (orthant)
            {
                xi_ = new double[size];
            }
        }
        else if (diag_.length != size || v_.length != size)
        {
            System.err.println("size of array is different");
            return -1;
        }
        else if (orthant && v_.length != size)
        {
            System.err.println("size of array is different");
            return -1;
        }
        int iflag = 0;
        if (orthant)
        {

            iflag = lbfgs_optimize(size,
                                   msize, x, f, g, diag_, w_, orthant, C, v_, xi_, iflag_);
            iflag_ = iflag;
        }
        else
        {
            iflag = lbfgs_optimize(size,
                                   msize, x, f, g, diag_, w_, orthant, C, g, xi_, iflag_);
            iflag_ = iflag;
        }

        if (iflag < 0)
        {
            System.err.println("routine stops with unexpected error");
            return -1;
        }

        if (iflag == 0)
        {
            clear();
            return 0;   // terminate
        }

        return 1;   // evaluate next f and g
    }
}
