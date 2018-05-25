package com.hankcs.hanlp.model.crf.crfpp;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * @author zhifac
 */
public abstract class FeatureIndex
{
    public static String[] BOS = {"_B-1", "_B-2", "_B-3", "_B-4", "_B-5", "_B-6", "_B-7", "_B-8"};
    public static String[] EOS = {"_B+1", "_B+2", "_B+3", "_B+4", "_B+5", "_B+6", "_B+7", "_B+8"};
    protected int maxid_;
    protected double[] alpha_;
    protected float[] alphaFloat_;
    protected double costFactor_;
    protected int xsize_;
    protected boolean checkMaxXsize_;
    protected int max_xsize_;
    protected int threadNum_;
    protected List<String> unigramTempls_;
    protected List<String> bigramTempls_;
    protected String templs_;
    protected List<String> y_;
    protected List<List<Path>> pathList_;
    protected List<List<Node>> nodeList_;

    public FeatureIndex()
    {
        maxid_ = 0;
        alpha_ = null;
        alphaFloat_ = null;
        costFactor_ = 1.0;
        xsize_ = 0;
        checkMaxXsize_ = false;
        max_xsize_ = 0;
        threadNum_ = 1;
        unigramTempls_ = new ArrayList<String>();
        bigramTempls_ = new ArrayList<String>();
        y_ = new ArrayList<String>();
    }

    protected abstract int getID(String s);

    /**
     * 计算状态特征函数的代价
     *
     * @param node
     */
    public void calcCost(Node node)
    {
        node.cost = 0.0;
        if (alphaFloat_ != null)
        {
            float c = 0.0f;
            for (int i = 0; node.fVector.get(i) != -1; i++)
            {
                c += alphaFloat_[node.fVector.get(i) + node.y];
            }
            node.cost = costFactor_ * c;
        }
        else
        {
            double c = 0.0;
            for (int i = 0; node.fVector.get(i) != -1; i++)
            {
                c += alpha_[node.fVector.get(i) + node.y];
            }
            node.cost = costFactor_ * c;
        }
    }

    /**
     * 计算转移特征函数的代价
     *
     * @param path 边
     */
    public void calcCost(Path path)
    {
        path.cost = 0.0;
        if (alphaFloat_ != null)
        {
            float c = 0.0f;
            for (int i = 0; path.fvector.get(i) != -1; i++)
            {
                c += alphaFloat_[path.fvector.get(i) + path.lnode.y * y_.size() + path.rnode.y];
            }
            path.cost = costFactor_ * c;
        }
        else
        {
            double c = 0.0;
            for (int i = 0; path.fvector.get(i) != -1; i++)
            {
                c += alpha_[path.fvector.get(i) + path.lnode.y * y_.size() + path.rnode.y];
            }
            path.cost = costFactor_ * c;
        }
    }

    public String makeTempls(List<String> unigramTempls, List<String> bigramTempls)
    {
        StringBuilder sb = new StringBuilder();
        for (String temp : unigramTempls)
        {
            sb.append(temp).append("\n");
        }
        for (String temp : bigramTempls)
        {
            sb.append(temp).append("\n");
        }
        return sb.toString();
    }

    public String getTemplate()
    {
        return templs_;
    }

    public String getIndex(String[] idxStr, int cur, TaggerImpl tagger)
    {
        int row = Integer.valueOf(idxStr[0]);
        int col = Integer.valueOf(idxStr[1]);
        int pos = row + cur;
        if (row < -EOS.length || row > EOS.length || col < 0 || col >= tagger.xsize())
        {
            return null;
        }

        //TODO(taku): very dirty workaround
        if (checkMaxXsize_)
        {
            max_xsize_ = Math.max(max_xsize_, col + 1);
        }
        if (pos < 0)
        {
            return BOS[-pos - 1];
        }
        else if (pos >= tagger.size())
        {
            return EOS[pos - tagger.size()];
        }
        else
        {
            return tagger.x(pos, col);
        }
    }

    public String applyRule(String str, int cur, TaggerImpl tagger)
    {
        StringBuilder sb = new StringBuilder();
        for (String tmp : str.split("%x", -1))
        {
            if (tmp.startsWith("U") || tmp.startsWith("B"))
            {
                sb.append(tmp);
            }
            else if (tmp.length() > 0)
            {
                String[] tuple = tmp.split("]");
                String[] idx = tuple[0].replace("[", "").split(",");
                String r = getIndex(idx, cur, tagger);
                if (r != null)
                {
                    sb.append(r);
                }
                if (tuple.length > 1)
                {
                    sb.append(tuple[1]);
                }
            }
        }

        return sb.toString();
    }

    private boolean buildFeatureFromTempl(List<Integer> feature, List<String> templs, int curPos, TaggerImpl tagger)
    {
        for (String tmpl : templs)
        {
            String featureID = applyRule(tmpl, curPos, tagger);
            if (featureID == null || featureID.length() == 0)
            {
                System.err.println("format error");
                return false;
            }
            int id = getID(featureID);
            if (id != -1)
            {
                feature.add(id);
            }
        }
        return true;
    }

    public boolean buildFeatures(TaggerImpl tagger)
    {
        List<Integer> feature = new ArrayList<Integer>();
        List<List<Integer>> featureCache = tagger.getFeatureCache_();
        tagger.setFeature_id_(featureCache.size());

        for (int cur = 0; cur < tagger.size(); cur++)
        {
            if (!buildFeatureFromTempl(feature, unigramTempls_, cur, tagger))
            {
                return false;
            }
            feature.add(-1);
            featureCache.add(feature);
            feature = new ArrayList<Integer>();
        }
        for (int cur = 1; cur < tagger.size(); cur++)
        {
            if (!buildFeatureFromTempl(feature, bigramTempls_, cur, tagger))
            {
                return false;
            }
            feature.add(-1);
            featureCache.add(feature);
            feature = new ArrayList<Integer>();
        }
        return true;
    }

    public void rebuildFeatures(TaggerImpl tagger)
    {
        int fid = tagger.getFeature_id_();
        List<List<Integer>> featureCache = tagger.getFeatureCache_();
        for (int cur = 0; cur < tagger.size(); cur++)
        {
            List<Integer> f = featureCache.get(fid++);
            for (int i = 0; i < y_.size(); i++)
            {
                Node n = new Node();
                n.clear();
                n.x = cur;
                n.y = i;
                n.fVector = f;
                tagger.set_node(n, cur, i);
            }
        }
        for (int cur = 1; cur < tagger.size(); cur++)
        {
            List<Integer> f = featureCache.get(fid++);
            for (int j = 0; j < y_.size(); j++)
            {
                for (int i = 0; i < y_.size(); i++)
                {
                    Path p = new Path();
                    p.clear();
                    p.add(tagger.node(cur - 1, j), tagger.node(cur, i));
                    p.fvector = f;
                }
            }
        }
    }

    public boolean open(String file)
    {
        return true;
    }

    public boolean open(InputStream stream)
    {
        return true;
    }

    public void clear()
    {

    }

    public int size()
    {
        return getMaxid_();
    }

    public int ysize()
    {
        return y_.size();
    }

    public int getMaxid_()
    {
        return maxid_;
    }

    public void setMaxid_(int maxid_)
    {
        this.maxid_ = maxid_;
    }

    public double[] getAlpha_()
    {
        return alpha_;
    }

    public void setAlpha_(double[] alpha_)
    {
        this.alpha_ = alpha_;
    }

    public float[] getAlphaFloat_()
    {
        return alphaFloat_;
    }

    public void setAlphaFloat_(float[] alphaFloat_)
    {
        this.alphaFloat_ = alphaFloat_;
    }

    public double getCostFactor_()
    {
        return costFactor_;
    }

    public void setCostFactor_(double costFactor_)
    {
        this.costFactor_ = costFactor_;
    }

    public int getXsize_()
    {
        return xsize_;
    }

    public void setXsize_(int xsize_)
    {
        this.xsize_ = xsize_;
    }

    public int getMax_xsize_()
    {
        return max_xsize_;
    }

    public void setMax_xsize_(int max_xsize_)
    {
        this.max_xsize_ = max_xsize_;
    }

    public int getThreadNum_()
    {
        return threadNum_;
    }

    public void setThreadNum_(int threadNum_)
    {
        this.threadNum_ = threadNum_;
    }

    public List<String> getUnigramTempls_()
    {
        return unigramTempls_;
    }

    public void setUnigramTempls_(List<String> unigramTempls_)
    {
        this.unigramTempls_ = unigramTempls_;
    }

    public List<String> getBigramTempls_()
    {
        return bigramTempls_;
    }

    public void setBigramTempls_(List<String> bigramTempls_)
    {
        this.bigramTempls_ = bigramTempls_;
    }

    public List<String> getY_()
    {
        return y_;
    }

    public void setY_(List<String> y_)
    {
        this.y_ = y_;
    }

    public List<List<Path>> getPathList_()
    {
        return pathList_;
    }

    public void setPathList_(List<List<Path>> pathList_)
    {
        this.pathList_ = pathList_;
    }

    public List<List<Node>> getNodeList_()
    {
        return nodeList_;
    }

    public void setNodeList_(List<List<Node>> nodeList_)
    {
        this.nodeList_ = nodeList_;
    }
}
