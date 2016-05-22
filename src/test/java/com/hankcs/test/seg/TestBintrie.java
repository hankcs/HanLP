package com.hankcs.test.seg;

import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.seg.Other.LongestBinSegmentToy;
import junit.framework.TestCase;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class TestBintrie extends TestCase {
	public void testPut() throws Exception {
		{
			BinTrie<String> binTrie = new BinTrie<String>();
			// binTrie.put("你好", "hello");
			// binTrie.put("我好", "fine");
			// binTrie.put("你坏", "bad");
			// System.out.println(binTrie.hasKey("我好"));
			// System.out.println(binTrie.get("我好"));
			// System.out.println(binTrie.hasKey("大家好"));
			// System.out.println(binTrie.get("大家好"));

			List<String> wordList = new ArrayList<String>();
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(
							"D:\\JavaProjects\\TireSpeed\\data\\WordList.txt")));
			String line;
			while ((line = br.readLine()) != null) {
				wordList.add(line);
				binTrie.put(line, "值：" + line);
			}
			br.close();

			// binTrie.remove("天才");
			// for (String w : wordList)
			// {
			// String value = binTrie.get(w);
			// if (!("值：" + w).equals(value))
			// {
			// throw new RuntimeException("错了：" + w + value);
			// }
			// // else
			// // {
			// // System.out.println(value);
			// // }
			// }

			// System.out.println(binTrie.hasKey("好孩子"));
			LongestBinSegmentToy<String> segmenter = new LongestBinSegmentToy<String>(
					binTrie);
			System.out.println(segmenter.seg("我们都是好孩子"));

			// Map.Entry<String, String> entry;
			// while ((entry = segmenter.next()) != null)
			// {
			// System.out.println(entry.getKey() + " - " +
			// (segmenter.getOffset() - entry.getKey().length()));
			// }
		}
	}

	// public void testSmartVsNormal() throws Exception
	// {
	// BinTrie<String> trieNormal = new BinTrie<>();
	// SmartBinTrie<String> trieSmart = new SmartBinTrie<>();
	// BinTrie<CoreDictionary.Attribute> dictionary =
	// CustomDictionary.getBinTrie();
	// List<String> wordList = new LinkedList<>();
	// for (Map.Entry<String, CoreDictionary.Attribute> entry :
	// dictionary.entrySet())
	// {
	// wordList.add(entry.getKey());
	// }
	// // 似乎内存会影响速度
	// {
	// BinTrie<String> trie = new BinTrie<>();
	// for (String word : wordList)
	// {
	// trie.put(word, word);
	// }
	// }
	//
	// long start;
	//
	// start = System.currentTimeMillis();
	// for (String word : wordList)
	// {
	// trieNormal.put(word, word);
	// }
	// System.out.printf("trieNormal首次插入耗时:%dms%n", System.currentTimeMillis() -
	// start);
	//
	// start = System.currentTimeMillis();
	// for (String word : wordList)
	// {
	// trieSmart.put(word, word);
	// }
	// System.out.printf("trieSmart首次插入耗时:%dms%n", System.currentTimeMillis() -
	// start);
	//
	// start = System.currentTimeMillis();
	// for (String word : wordList)
	// {
	// trieNormal.put(word, word);
	// }
	// System.out.printf("trieNormal再次插入耗时:%dms%n", System.currentTimeMillis() -
	// start);
	//
	// start = System.currentTimeMillis();
	// for (String word : wordList)
	// {
	// trieSmart.put(word, word);
	// }
	// System.out.printf("trieSmart再次插入耗时:%dms%n", System.currentTimeMillis() -
	// start);
	//
	// start = System.currentTimeMillis();
	// for (String word : wordList)
	// {
	// assertEquals(word, trieNormal.get(word));
	// }
	// System.out.printf("trieNormal查询耗时:%dms%n", System.currentTimeMillis() -
	// start);
	//
	// start = System.currentTimeMillis();
	// for (String word : wordList)
	// {
	// assertEquals(word, trieSmart.get(word));
	// }
	// System.out.printf("trieSmart查询耗时:%dms%n", System.currentTimeMillis() -
	// start);
	// }
}
