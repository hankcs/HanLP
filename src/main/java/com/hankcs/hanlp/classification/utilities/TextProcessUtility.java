package com.hankcs.hanlp.classification.utilities;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NotionalTokenizer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;

/**
 * 文件预处理工具
 */
public class TextProcessUtility
{
    /**
     * 预处理，去除标点，空格和停用词
     *
     * @param text
     * @return
     */
    public static String preprocess(String text)
    {
        return text.replaceAll("\\p{P}", " ").replaceAll("\\s+", " ").toLowerCase(Locale.getDefault());
    }

    /**
     * 提取关键词，在真实的应用场景中，还应该涉及到短语
     *
     * @param text
     * @return
     */
    public static String[] extractKeywords(String text)
    {
        List<Term> termList = NotionalTokenizer.segment(text);
        String[] wordArray = new String[termList.size()];
        Iterator<Term> iterator = termList.iterator();
        for (int i = 0; i < wordArray.length; i++)
        {
            wordArray[i] = iterator.next().word;
        }
        return wordArray;
    }

    /**
     * 统计每个词的词频
     *
     * @param keywordArray
     * @return
     */
    public static Map<String, Integer> getKeywordCounts(String[] keywordArray)
    {
        Map<String, Integer> counts = new HashMap<String, Integer>();

        Integer counter;
        for (int i = 0; i < keywordArray.length; ++i)
        {
            counter = counts.get(keywordArray[i]);
            if (counter == null)
            {
                counter = 0;
            }
            counts.put(keywordArray[i], ++counter); //增加词频
        }

        return counts;
    }

    /**
     * 加载一个文件夹下的所有语料
     *
     * @param path
     * @return
     */
    public static Map<String, String[]> loadCorpus(String path)
    {
        Map<String, String[]> dataSet = new TreeMap<String, String[]>();
        File root = new File(path);
        File[] folders = root.listFiles();
        if (folders == null) return null;
        for (File folder : folders)
        {
            if (folder.isFile()) continue;
            File[] files = folder.listFiles();
            if (files == null) continue;
            String[] documents = new String[files.length];
            for (int i = 0; i < files.length; i++)
            {
                documents[i] = IOUtil.readTxt(files[i].getAbsolutePath());
            }
            dataSet.put(folder.getName(), documents);
        }

        return dataSet;
    }

    /**
     * 加载一个文件夹下的所有语料
     *
     * @param folderPath
     * @return
     */
    public static Map<String, String[]> loadCorpusWithException(String folderPath, String charsetName) throws IOException
    {
        if (folderPath == null) throw new IllegalArgumentException("参数 folderPath == null");
        File root = new File(folderPath);
        if (!root.exists()) throw new IllegalArgumentException(String.format("目录 %s 不存在", root.getAbsolutePath()));
        if (!root.isDirectory())
            throw new IllegalArgumentException(String.format("目录 %s 不是一个目录", root.getAbsolutePath()));

        Map<String, String[]> dataSet = new TreeMap<String, String[]>();
        File[] folders = root.listFiles();
        if (folders == null) return null;
        for (File folder : folders)
        {
            if (folder.isFile()) continue;
            File[] files = folder.listFiles();
            if (files == null) continue;
            String[] documents = new String[files.length];
            for (int i = 0; i < files.length; i++)
            {
                documents[i] = readTxt(files[i], charsetName);
            }
            dataSet.put(folder.getName(), documents);
        }

        return dataSet;
    }

    public static String readTxt(File file, String charsetName) throws IOException
    {
        FileInputStream is = new FileInputStream(file);
        byte[] targetArray = new byte[is.available()];
        int len;
        int off = 0;
        while ((len = is.read(targetArray, off, targetArray.length - off)) != -1 && off < targetArray.length)
        {
            off += len;
        }
        is.close();

        return new String(targetArray, charsetName);
    }

    public static Map<String, String[]> loadCorpusWithException(String corpusPath) throws IOException
    {
        return loadCorpusWithException(corpusPath, "UTF-8");
    }
}