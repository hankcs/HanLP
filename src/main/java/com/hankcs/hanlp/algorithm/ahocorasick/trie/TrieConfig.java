package com.hankcs.hanlp.algorithm.ahocorasick.trie;

/**
 * 配置
 */
public class TrieConfig
{
    /**
     * 允许重叠
     */
    private boolean allowOverlaps = true;

    /**
     * 只保留最长匹配
     */
    public boolean remainLongest = false;

    /**
     * 是否允许重叠
     *
     * @return
     */
    public boolean isAllowOverlaps()
    {
        return allowOverlaps;
    }

    /**
     * 设置是否允许重叠
     *
     * @param allowOverlaps
     */
    public void setAllowOverlaps(boolean allowOverlaps)
    {
        this.allowOverlaps = allowOverlaps;
    }
}
