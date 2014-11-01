package com.hankcs.hanlp.algoritm.ahocorasick.trie;

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
     * 只匹配完整单词
     */
    private boolean onlyWholeWords = false;

    /**
     * 大小写不敏感
     */
    private boolean caseInsensitive = false;

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

    /**
     * 是否只匹配完整单词
     *
     * @return
     */
    public boolean isOnlyWholeWords()
    {
        return onlyWholeWords;
    }

    /**
     * 设置是否只匹配完整单词
     *
     * @param onlyWholeWords
     */
    public void setOnlyWholeWords(boolean onlyWholeWords)
    {
        this.onlyWholeWords = onlyWholeWords;
    }

    /**
     * 是否大小写敏感
     *
     * @return
     */
    public boolean isCaseInsensitive()
    {
        return caseInsensitive;
    }

    /**
     * 设置大小写敏感
     *
     * @param caseInsensitive
     */
    public void setCaseInsensitive(boolean caseInsensitive)
    {
        this.caseInsensitive = caseInsensitive;
    }
}
