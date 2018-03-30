package com.hankcs.hanlp.model.crf.crfpp;

/**
 * @author zhifac
 */
public class Mcsrch
{
    public static final double ftol = 1e-4;
    public static final double xtol = 1e-16;
    public static final double eps = 1e-7;
    public static final double lb3_1_gtol = 0.9;
    public static final double lb3_1_stpmin = 1e-20;
    public static final double lb3_1_stpmax = 1e20;
    public static final int lb3_1_mp = 6;
    public static final int lb3_1_lp = 6;

    int infoc;
    boolean stage1, brackt;
    double finit, dginit, dgtest, width, width1;
    double stx, fx, dgx, sty, fy, dgy, stmin, stmax;

    public Mcsrch()
    {
        infoc = 0;
        stage1 = brackt = false;
        finit = dginit = dgtest = width = width1 = 0.0;
        stx = fx = dgx = sty = fy = dgy = stmin = stmax = 0.0;
    }

    public static double sigma(double x)
    {
        if (x > 0) return 1.0;
        else if (x < 0) return -1.0;
        return 0.0;
    }

    public double pi(double x, double y)
    {
        return sigma(x) == sigma(y) ? x : 0.0;
    }

    public static void daxpy_(int n, double da, double[] dx, int offsetX, double[] dy, int offsetY)
    {
        for (int i = 0; i < n; ++i)
            dy[i + offsetY] += da * dx[i + offsetX];
    }

    public static double ddot_(int size, double[] dx, int offsetX, double[] dy, int offsetY)
    {
        double res = 0.0;
        for (int i = 0; i < size; i++)
        {
            res += dx[i + offsetX] * dy[i + offsetY];
        }
        return res;
    }

    public static void mcstep(double[] stx, double[] fx, double[] dx,
                              double[] sty, double[] fy, double[] dy,
                              double[] stp, double fp, double dp,
                              boolean[] brackt,
                              double stpmin, double stpmax,
                              int[] info)
    {
        boolean bound = true;
        double p, q, s, d1, d2, d3, r, gamma, theta, stpq, stpc, stpf;
        info[0] = 0;

        if (brackt[0] && ((stp[0] <= Math.min(stx[0], sty[0]) || stp[0] >= Math.max(stx[0], sty[0])) ||
                dx[0] * (stp[0] - stx[0]) >= 0.0 || stpmax < stpmin))
        {
            return;
        }

        double sgnd = dp * (dx[0] / Math.abs(dx[0]));

        if (fp > fx[0])
        {
            info[0] = 1;
            bound = true;
            theta = (fx[0] - fp) * 3 / (stp[0] - stx[0]) + dx[0] + dp;
            d1 = Math.abs(theta);
            d2 = Math.abs(dx[0]);
            d1 = Math.max(d1, d2);
            d2 = Math.abs(dp);
            s = Math.max(d1, d2);
            d1 = theta / s;
            gamma = s * Math.sqrt(d1 * d1 - dx[0] / s * (dp / s));
            if (stp[0] < stx[0])
            {
                gamma = -gamma;
            }
            p = gamma - dx[0] + theta;
            q = gamma - dx[0] + gamma + dp;
            r = p / q;
            stpc = stx[0] + r * (stp[0] - stx[0]);
            stpq = stx[0] + dx[0] / ((fx[0] - fp) /
                    (stp[0] - stx[0]) + dx[0]) / 2 * (stp[0] - stx[0]);
            d1 = stpc - stx[0];
            d2 = stpq - stx[0];
            if (Math.abs(d1) < Math.abs(d2))
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpc + (stpq - stpc) / 2;
            }
            brackt[0] = true;
        }
        else if (sgnd < 0.0)
        {
            info[0] = 2;
            bound = false;
            theta = (fx[0] - fp) * 3 / (stp[0] - stx[0]) + dx[0] + dp;
            d1 = Math.abs(theta);
            d2 = Math.abs(dx[0]);
            d1 = Math.max(d1, d2);
            d2 = Math.abs(dp);
            s = Math.max(d1, d2);
            d1 = theta / s;
            gamma = s * Math.sqrt(d1 * d1 - dx[0] / s * (dp / s));
            if (stp[0] > stx[0])
            {
                gamma = -gamma;
            }
            p = gamma - dp + theta;
            q = gamma - dp + gamma + dx[0];
            r = p / q;
            stpc = stp[0] + r * (stx[0] - stp[0]);
            stpq = stp[0] + dp / (dp - dx[0]) * (stx[0] - stp[0]);
            d1 = stpc - stp[0];
            d2 = stpq - stp[0];
            if (Math.abs(d1) > Math.abs(d2))
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
            brackt[0] = true;
        }
        else if (Math.abs(dp) < Math.abs(dx[0]))
        {
            info[0] = 3;
            bound = true;
            theta = (fx[0] - fp) * 3 / (stp[0] - stx[0]) + dx[0] + dp;
            d1 = Math.abs(theta);
            d2 = Math.abs(dx[0]);
            d1 = Math.max(d1, d2);
            d2 = Math.abs(dp);
            s = Math.max(d1, d2);
            d3 = theta / s;
            d1 = 0.0;
            d2 = d3 * d3 - dx[0] / s * (dp / s);
            gamma = s * Math.sqrt((Math.max(d1, d2)));
            if (stp[0] > stx[0])
            {
                gamma = -gamma;
            }
            p = gamma - dp + theta;
            q = gamma + (dx[0] - dp) + gamma;
            r = p / q;
            if (r < 0.0 && gamma != 0.0)
            {
                stpc = stp[0] + r * (stx[0] - stp[0]);
            }
            else if (stp[0] > stx[0])
            {
                stpc = stpmax;
            }
            else
            {
                stpc = stpmin;
            }
            stpq = stp[0] + dp / (dp - dx[0]) * (stx[0] - stp[0]);
            if (brackt[0])
            {
                d1 = stp[0] - stpc;
                d2 = stp[0] - stpq;
                if (Math.abs(d1) < Math.abs(d2))
                {
                    stpf = stpc;
                }
                else
                {
                    stpf = stpq;
                }
            }
            else
            {
                d1 = stp[0] - stpc;
                d2 = stp[0] - stpq;
                if (Math.abs(d1) > Math.abs(d2))
                {
                    stpf = stpc;
                }
                else
                {
                    stpf = stpq;
                }
            }
        }
        else
        {
            info[0] = 4;
            bound = false;
            if (brackt[0])
            {
                theta = (fp - fy[0]) * 3 / (sty[0] - stp[0]) + dy[0] + dp;
                d1 = Math.abs(theta);
                d2 = Math.abs(dy[0]);
                d1 = Math.max(d1, d2);
                d2 = Math.abs(dp);
                s = Math.max(d1, d2);
                d1 = theta / s;
                gamma = s * Math.sqrt(d1 * d1 - dy[0] / s * (dp / s));
                if (stp[0] > sty[0])
                {
                    gamma = -gamma;
                }
                p = gamma - dp + theta;
                q = gamma - dp + gamma + dy[0];
                r = p / q;
                stpc = stp[0] + r * (sty[0] - stp[0]);
                stpf = stpc;
            }
            else if (stp[0] > stx[0])
            {
                stpf = stpmax;
            }
            else
            {
                stpf = stpmin;
            }
        }

        if (fp > fx[0])
        {
            sty[0] = stp[0];
            fy[0] = fp;
            dy[0] = dp;
        }
        else
        {
            if (sgnd < 0.0)
            {
                sty[0] = stx[0];
                fy[0] = fx[0];
                dy[0] = dx[0];
            }
            stx[0] = stp[0];
            fx[0] = fp;
            dx[0] = dp;
        }

        stpf = Math.min(stpmax, stpf);
        stpf = Math.max(stpmin, stpf);
        stp[0] = stpf;
        if (brackt[0] && bound)
        {
            if (sty[0] > stx[0])
            {
                d1 = stx[0] + (sty[0] - stx[0]) * 0.66;
                stp[0] = Math.min(d1, stp[0]);
            }
            else
            {
                d1 = stx[0] + (sty[0] - stx[0]) * 0.66;
                stp[0] = Math.max(d1, stp[0]);
            }
        }

        return;
    }


    void mcsrch(int size,
                double[] x,
                double f, double[] g, double[] s, int startOffset,
                double[] stp,
                int[] info, int[] nfev, double[] wa)
    {
        double p5 = 0.5;
        double p66 = 0.66;
        double xtrapf = 4.0;
        int maxfev = 20;

        if (info[0] != -1)
        {
            infoc = 1;

            if (size <= 0 || stp[0] <= 0.0)
            {
                return;
            }

            dginit = ddot_(size, g, 0, s, startOffset);
            if (dginit >= 0.0) return;

            brackt = false;
            stage1 = true;
            nfev[0] = 0;
            finit = f;
            dgtest = ftol * dginit;
            width = lb3_1_stpmax - lb3_1_stpmin;
            width1 = width / p5;
            for (int j = 0; j < size; ++j)
            {
                wa[j] = x[j];
            }

            stx = 0.0;
            fx = finit;
            dgx = dginit;
            sty = 0.0;
            fy = finit;
            dgy = dginit;
        }

        boolean firstLoop = true;
        while (true)
        {
            if (!firstLoop || (firstLoop && info[0] != -1))
            {
                if (brackt)
                {
                    stmin = Math.min(stx, sty);
                    stmax = Math.max(stx, sty);
                }
                else
                {
                    stmin = stx;
                    stmax = stp[0] + xtrapf * (stp[0] - stx);
                }

                stp[0] = Math.max(stp[0], lb3_1_stpmin);
                stp[0] = Math.min(stp[0], lb3_1_stpmax);

                if ((brackt && ((stp[0] <= stmin || stp[0] >= stmax) ||
                        nfev[0] >= maxfev - 1 || infoc == 0)) ||
                        (brackt && (stmax - stmin <= xtol * stmax)))
                {
                    stp[0] = stx;
                }

                for (int j = 0; j < size; ++j)
                {
                    x[j] = wa[j] + stp[0] * s[startOffset + j];
                }
                info[0] = -1;
                return;
            }

            info[0] = 0;
            ++(nfev[0]);
            double dg = ddot_(size, g, 0, s, startOffset);
            double ftest1 = finit + stp[0] * dgtest;

            if (brackt && ((stp[0] <= stmin || stp[0] >= stmax) || infoc == 0))
            {
                info[0] = 6;
            }
            if (stp[0] == lb3_1_stpmax && f <= ftest1 && dg <= dgtest)
            {
                info[0] = 5;
            }
            if (stp[0] == lb3_1_stpmin && (f > ftest1 || dg >= dgtest))
            {
                info[0] = 4;
            }
            if (nfev[0] >= maxfev)
            {
                info[0] = 3;
            }
            if (brackt && stmax - stmin <= xtol * stmax)
            {
                info[0] = 2;
            }
            if (f <= ftest1 && Math.abs(dg) <= lb3_1_gtol * (-dginit))
            {
                info[0] = 1;
            }
            if (info[0] != 0)
            {
                return;
            }

            if (stage1 && f <= ftest1 && dg >= Math.min(ftol, lb3_1_gtol) * dginit)
            {
                stage1 = false;
            }

            if (stage1 && f <= fx && f > ftest1)
            {
                double fm = f - stp[0] * dgtest;
                double fxm = fx - stx * dgtest;
                double fym = fy - sty * dgtest;
                double dgm = dg - dgtest;
                double dgxm = dgx - dgtest;
                double dgym = dgy - dgtest;

                double[] stxArr = {stx};
                double[] fxmArr = {fxm};
                double[] dgxmArr = {dgxm};
                double[] styArr = {sty};
                double[] fymArr = {fym};
                double[] dgymArr = {dgym};
                boolean[] bracktArr = {brackt};
                int[] infocArr = {infoc};
                mcstep(stxArr, fxmArr, dgxmArr, styArr, fymArr, dgymArr, stp, fm, dgm, bracktArr,
                       stmin, stmax, infocArr);
                stx = stxArr[0];
                fxm = fxmArr[0];
                dgxm = dgxmArr[0];
                sty = styArr[0];
                fym = fymArr[0];
                dgym = dgymArr[0];
                brackt = bracktArr[0];
                infoc = infocArr[0];

                fx = fxm + stx * dgtest;
                fy = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest;
            }
            else
            {
                double[] stxArr = {stx};
                double[] fxArr = {fx};
                double[] dgxArr = {dgx};
                double[] styArr = {sty};
                double[] fyArr = {fy};
                double[] dgyArr = {dgy};
                boolean[] bracktArr = {brackt};
                int[] infocArr = {infoc};
                mcstep(stxArr, fxArr, dgxArr, styArr, fyArr, dgyArr, stp, f, dg, bracktArr,
                       stmin, stmax, infocArr);
                stx = stxArr[0];
                fx = fxArr[0];
                dgx = dgxArr[0];
                sty = styArr[0];
                fy = fyArr[0];
                dgy = dgyArr[0];
                brackt = bracktArr[0];
                infoc = infocArr[0];
            }

            if (brackt)
            {
                double d1 = sty - stx;
                if (Math.abs(d1) >= p66 * width1)
                {
                    stp[0] = stx + p5 * (sty - stx);
                }
                width1 = width;
                d1 = sty - stx;
                width = Math.abs(d1);
            }
            firstLoop = false;
        }
    }
}
