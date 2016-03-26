/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 23:04</create-date>
 *
 * <copyright file="Util.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;


import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 一些常用的IO操作
 *
 * @author hankcs
 */
public class IOUtil
{
    /**
     * 序列化对象
     *
     * @param o
     * @param path
     * @return
     */
    public static boolean saveObjectTo(Object o, String path)
    {
        try
        {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
            oos.writeObject(o);
            oos.close();
        }
        catch (IOException e)
        {
            logger.warning("在保存对象" + o + "到" + path + "时发生异常" + e);
            return false;
        }

        return true;
    }

    /**
     * 反序列化对象
     *
     * @param path
     * @return
     */
    public static Object readObjectFrom(String path)
    {
        ObjectInputStream ois = null;
        try
        {
            ois = new ObjectInputStream(new FileInputStream(path));
            Object o = ois.readObject();
            ois.close();
            return o;
        }
        catch (Exception e)
        {
            logger.warning("在从" + path + "读取对象时发生异常" + e);
        }

        return null;
    }

    /**
     * 一次性读入纯文本
     *
     * @param path
     * @return
     */
    public static String readTxt(String path)
    {
        if (path == null) return null;
        File file = new File(path);
        Long fileLength = file.length();
        byte[] fileContent = new byte[fileLength.intValue()];
        try
        {
            FileInputStream in = new FileInputStream(file);
            in.read(fileContent);
            in.close();
        }
        catch (FileNotFoundException e)
        {
            logger.warning("找不到" + path + e);
            return null;
        }
        catch (IOException e)
        {
            logger.warning("读取" + path + "发生IO异常" + e);
            return null;
        }

        return new String(fileContent, Charset.forName("UTF-8"));
    }

    public static LinkedList<String[]> readCsv(String path)
    {
        LinkedList<String[]> resultList = new LinkedList<String[]>();
        LinkedList<String> lineList = readLineList(path);
        for (String line : lineList)
        {
            resultList.add(line.split(","));
        }
        return resultList;
    }

    /**
     * 快速保存
     *
     * @param path
     * @param content
     * @return
     */
    public static boolean saveTxt(String path, String content)
    {
        try
        {
            FileChannel fc = new FileOutputStream(path).getChannel();
            fc.write(ByteBuffer.wrap(content.getBytes()));
            fc.close();
        }
        catch (Exception e)
        {
            logger.throwing("IOUtil", "saveTxt", e);
            logger.warning("IOUtil saveTxt 到" + path + "失败" + e.toString());
            return false;
        }
        return true;
    }

    public static boolean saveTxt(String path, StringBuilder content)
    {
        return saveTxt(path, content.toString());
    }

    public static <T> boolean saveCollectionToTxt(Collection<T> collection, String path)
    {
        StringBuilder sb = new StringBuilder();
        for (Object o : collection)
        {
            sb.append(o);
            sb.append('\n');
        }
        return saveTxt(path, sb.toString());
    }

    /**
     * 将整个文件读取为字节数组
     *
     * @param path
     * @return
     */
    public static byte[] readBytes(String path)
    {
        try
        {
            FileInputStream fis = new FileInputStream(path);
            FileChannel channel = fis.getChannel();
            int fileSize = (int) channel.size();
            ByteBuffer byteBuffer = ByteBuffer.allocate(fileSize);
            channel.read(byteBuffer);
            byteBuffer.flip();
            byte[] bytes = byteBuffer.array();
            byteBuffer.clear();
            channel.close();
            fis.close();
            return bytes;
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "时发生异常" + e);
        }

        return null;
    }

    public static LinkedList<String> readLineList(String path)
    {
        LinkedList<String> result = new LinkedList<String>();
        String txt = readTxt(path);
        if (txt == null) return result;
        StringTokenizer tokenizer = new StringTokenizer(txt, "\n");
        while (tokenizer.hasMoreTokens())
        {
            result.add(tokenizer.nextToken());
        }

        return result;
    }

    /**
     * 用省内存的方式读取大文件
     *
     * @param path
     * @return
     */
    public static LinkedList<String> readLineListWithLessMemory(String path)
    {
        LinkedList<String> result = new LinkedList<String>();
        String line = null;
        try
        {
            BufferedReader bw = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
            while ((line = bw.readLine()) != null)
            {
                result.add(line);
            }
            bw.close();
        }
        catch (Exception e)
        {
            logger.warning("加载" + path + "失败，" + e);
        }

        return result;
    }

    public static boolean saveMapToTxt(Map<Object, Object> map, String path)
    {
        return saveMapToTxt(map, path, "=");
    }

    public static boolean saveMapToTxt(Map<Object, Object> map, String path, String separator)
    {
        map = new TreeMap<Object, Object>(map);
        return saveEntrySetToTxt(map.entrySet(), path, separator);
    }

    public static boolean saveEntrySetToTxt(Set<Map.Entry<Object, Object>> entrySet, String path, String separator)
    {
        StringBuilder sbOut = new StringBuilder();
        for (Map.Entry<Object, Object> entry : entrySet)
        {
            sbOut.append(entry.getKey());
            sbOut.append(separator);
            sbOut.append(entry.getValue());
            sbOut.append('\n');
        }
        return saveTxt(path, sbOut.toString());
    }

    /**
     * 获取文件所在目录的路径
     * @param path
     * @return
     */
    public static String dirname(String path)
    {
        int index = path.lastIndexOf('/');
        if (index == -1) return path;
        return path.substring(0, index + 1);
    }

    public static LineIterator readLine(String path)
    {
        return new LineIterator(path);
    }

    /**
     * 方便读取按行读取大文件
     */
    public static class LineIterator implements Iterator<String>
    {
        BufferedReader bw;
        String line;

        public LineIterator(String path)
        {
            try
            {
                bw = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
                line = bw.readLine();
            }
            catch (FileNotFoundException e)
            {
                logger.warning("文件" + path + "不存在，接下来的调用会返回null" + TextUtility.exceptionToString(e));
                bw = null;
            }
            catch (IOException e)
            {
                logger.warning("在读取过程中发生错误" + TextUtility.exceptionToString(e));
                bw = null;
            }
        }

        public void close()
        {
            if (bw == null) return;
            try
            {
                bw.close();
                bw = null;
            }
            catch (IOException e)
            {
                logger.warning("关闭文件失败" + TextUtility.exceptionToString(e));
            }
            return;
        }

        @Override
        public boolean hasNext()
        {
            if (bw == null) return false;
            if (line == null)
            {
                try
                {
                    bw.close();
                    bw = null;
                }
                catch (IOException e)
                {
                    logger.warning("关闭文件失败" + TextUtility.exceptionToString(e));
                }
                return false;
            }

            return true;
        }

        @Override
        public String next()
        {
            String preLine = line;
            try
            {
                if (bw != null)
                {
                    line = bw.readLine();
                    if (line == null && bw != null)
                    {
                        try
                        {
                            bw.close();
                            bw = null;
                        }
                        catch (IOException e)
                        {
                            logger.warning("关闭文件失败" + TextUtility.exceptionToString(e));
                        }
                    }
                }
                else
                {
                    line = null;
                }
            }
            catch (IOException e)
            {
                logger.warning("在读取过程中发生错误" + TextUtility.exceptionToString(e));
            }
            return preLine;
        }

        @Override
        public void remove()
        {
            throw new UnsupportedOperationException("只读，不可写！");
        }
    }

    /**
     * 创建一个BufferedWriter
     *
     * @param path
     * @return
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    public static BufferedWriter newBufferedWriter(String path) throws FileNotFoundException, UnsupportedEncodingException
    {
        return new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));
    }

    /**
     * 创建一个BufferedReader
     * @param path
     * @return
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    public static BufferedReader newBufferedReader(String path) throws FileNotFoundException, UnsupportedEncodingException
    {
        return new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
    }

    public static BufferedWriter newBufferedWriter(String path, boolean append) throws FileNotFoundException, UnsupportedEncodingException
    {
        return new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path, append), "UTF-8"));
    }

    /**
     * 获取最后一个分隔符的后缀
     * @param name
     * @param delimiter
     * @return
     */
    public static String getSuffix(String name, String delimiter)
    {
        return name.substring(name.lastIndexOf(delimiter) + 1);
    }

    /**
     * 写数组，用制表符分割
     * @param bw
     * @param params
     * @throws IOException
     */
    public static void writeLine(BufferedWriter bw, String... params) throws IOException
    {
        for (int i = 0; i < params.length - 1; i++)
        {
            bw.write(params[i]);
            bw.write('\t');
        }
        bw.write(params[params.length - 1]);
    }

    /**
     * 加载词典，词典必须遵守HanLP核心词典格式
     * @param pathArray 词典路径，可以有任意个
     * @return 一个储存了词条的map
     * @throws IOException 异常表示加载失败
     */
    public static TreeMap<String, CoreDictionary.Attribute> loadDictionary(String... pathArray) throws IOException
    {
        TreeMap<String, CoreDictionary.Attribute> map = new TreeMap<String, CoreDictionary.Attribute>();
        for (String path : pathArray)
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
            loadDictionary(br, map);
        }

        return map;
    }

    /**
     * 将一个BufferedReader中的词条加载到词典
     * @param br 源
     * @param storage 储存位置
     * @throws IOException 异常表示加载失败
     */
    public static void loadDictionary(BufferedReader br, TreeMap<String, CoreDictionary.Attribute> storage) throws IOException
    {
        String line;
        while ((line = br.readLine()) != null)
        {
            String param[] = line.split("\\s");
            int natureCount = (param.length - 1) / 2;
            CoreDictionary.Attribute attribute = new CoreDictionary.Attribute(natureCount);
            for (int i = 0; i < natureCount; ++i)
            {
                attribute.nature[i] = Enum.valueOf(Nature.class, param[1 + 2 * i]);
                attribute.frequency[i] = Integer.parseInt(param[2 + 2 * i]);
                attribute.totalFrequency += attribute.frequency[i];
            }
            storage.put(param[0], attribute);
        }
        br.close();
    }

    public static void writeCustomNature(DataOutputStream out, LinkedHashSet<Nature> customNatureCollector) throws IOException
    {
        if (customNatureCollector.size() == 0) return;
        out.writeInt(-customNatureCollector.size());
        for (Nature nature : customNatureCollector)
        {
            TextUtility.writeString(nature.toString(), out);
        }
    }
}
