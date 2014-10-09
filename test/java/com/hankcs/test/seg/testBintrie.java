package com.hankcs.test.seg;


import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.seg.LongestSegmenter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class testBintrie
{
    public static void main(String[] args) throws IOException
    {
        BinTrie<String> binTrie = new BinTrie<String>();
//        binTrie.put("你好", "hello");
//        binTrie.put("我好", "fine");
//        binTrie.put("你坏", "bad");
//        System.out.println(binTrie.hasKey("我好"));
//        System.out.println(binTrie.get("我好"));
//        System.out.println(binTrie.hasKey("大家好"));
//        System.out.println(binTrie.get("大家好"));

        List<String> wordList = new ArrayList<String>();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("D:\\JavaProjects\\TireSpeed\\data\\WordList.txt")));
        String line;
        while ((line = br.readLine()) != null)
        {
            wordList.add(line);
            binTrie.put(line, "值：" + line);
        }
        br.close();

//        binTrie.remove("天才");
//        for (String w : wordList)
//        {
//            String value = binTrie.get(w);
//            if (!("值：" + w).equals(value))
//            {
//                throw new RuntimeException("错了：" + w + value);
//            }
////            else
////            {
////                System.out.println(value);
////            }
//        }

//        System.out.println(binTrie.hasKey("好孩子"));
        LongestSegmenter<String> segmenter = new LongestSegmenter<String>(binTrie);
        System.out.println(segmenter.seg("我们都是好孩子"));

//        Map.Entry<String, String> entry;
//        while ((entry = segmenter.next()) != null)
//        {
//            System.out.println(entry.getKey() + " - " + (segmenter.getOffset() - entry.getKey().length()));
//        }
    }
}
