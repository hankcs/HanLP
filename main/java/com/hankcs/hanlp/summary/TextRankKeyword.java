package com.hankcs.hanlp.summary;


import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.NShort.Path.WordResult;
import com.hankcs.hanlp.seg.NShort.Segment;

import java.util.*;

/**
 * 基于TextRank算法的关键字提取，适用于单文档
 * @author hankcs
 */
public class TextRankKeyword extends KeywordExtractor
{
    /**
     * 提取多少个关键字
     */
    public int nKeyword = 10;
    /**
     * 阻尼系数（ＤａｍｐｉｎｇＦａｃｔｏｒ），一般取值为0.85
     */
    final float d = 0.85f;
    /**
     * 最大迭代次数
     */
    final int max_iter = 200;
    final float min_diff = 0.001f;

    CoreStopWordDictionary stopWordDictionary;

    public TextRankKeyword()
    {
        // jdk bug : Exception in thread "main" java.lang.IllegalArgumentException: Comparison method violates its general contract!
        System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
    }

    public String[] getKeyword(String title, String content)
    {
        List<WordResult> termList = Segment.parse(title + content);
        List<String> wordList = new ArrayList<String>();
        for (WordResult t : termList)
        {
            if (shouldInclude(t))
            {
                wordList.add(t.sWord);
            }
        }
//        System.out.println(wordList);
        BinTrie<BinTrie<Boolean>> words = new BinTrie<BinTrie<Boolean>>();
        Queue<String> que = new LinkedList<String>();
        for (String w : wordList)
        {
            if (!words.containsKey(w))
            {
                words.put(w, new BinTrie<Boolean>());
            }
            que.offer(w);
            if (que.size() > 5)
            {
                que.poll();
            }

            for (String w1 : que)
            {
                for (String w2 : que)
                {
                    if (w1.equals(w2))
                    {
                        continue;
                    }

                    words.get(w1).put(w2, true);
                    words.get(w2).put(w1, true);
                }
            }
        }
//        System.out.println(words);
        Map<String, Float> score = new HashMap<String, Float>();
        for (int i = 0; i < max_iter; ++i)
        {
            Map<String, Float> m = new HashMap<String, Float>();
            float max_diff = 0;
            for (Map.Entry<String, BinTrie<Boolean>> entry : words.entrySet())
            {
                String key = entry.getKey();
                BinTrie<Boolean> value = entry.getValue();
                m.put(key, 1 - d);
                Set<Map.Entry<String, Boolean>> set = value.entrySet();
                for (Map.Entry<String, Boolean> element : set)
                {
                    String other = element.getKey();
                    int size = words.get(other).size();
                    if (key.equals(other) || size == 0) continue;
                    m.put(key, m.get(key) + d / size * (score.get(other) == null ? 0 : score.get(other)));
                }
                max_diff = Math.max(max_diff, Math.abs(m.get(key) - (score.get(key) == null ? 0 : score.get(key))));
            }
            score = m;
            if (max_diff <= min_diff) break;
        }
        List<Map.Entry<String, Float>> entryList = new ArrayList<Map.Entry<String, Float>>(score.entrySet());
        Collections.sort(entryList, new Comparator<Map.Entry<String, Float>>()
        {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2)
            {
                return (o1.getValue() - o2.getValue() > 0 ? -1 : 1);
            }
        });
//        System.out.println(entryList);
        int limit = Math.min(nKeyword, entryList.size());
        String[] result = new String[limit];
        for (int i = 0; i < limit; ++i)
        {
            result[i] = entryList.get(i).getKey();
        }
        return result;
    }

}
