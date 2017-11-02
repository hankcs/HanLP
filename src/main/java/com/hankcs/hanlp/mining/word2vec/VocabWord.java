package com.hankcs.hanlp.mining.word2vec;

public class VocabWord implements Comparable<VocabWord>
{

    public static final int MAX_CODE_LENGTH = 40;

    int cn, codelen;
    int[] point;
    String word;
    char[] code;

    public VocabWord(String word)
    {
        this.word = word;
        cn = 0;
        point = new int[MAX_CODE_LENGTH];
        code = new char[MAX_CODE_LENGTH];
    }

    public void setCn(int cn)
    {
        this.cn = cn;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[%s] cn=%d, codelen=%d, ", word, cn, codelen));
        sb.append("code=(");
        for (int i = 0; i < codelen; i++)
        {
            if (i > 0) sb.append(',');
            sb.append(code[i]);
        }
        sb.append("), point=(");
        for (int i = 0; i < codelen; i++)
        {
            if (i > 0) sb.append(',');
            sb.append(point[i]);
        }
        sb.append(")");
        return sb.toString();
    }

    @Override
    public int compareTo(VocabWord that)
    {
        return that.cn - this.cn;
    }
}
