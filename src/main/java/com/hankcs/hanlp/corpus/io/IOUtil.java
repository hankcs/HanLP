
package com.hankcs.hanlp.corpus.io;

import com.hankcs.hanlp.HanLP.Config;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary.Attribute;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Map.Entry;

public class IOUtil {
    public IOUtil() {
    }

    public static boolean saveObjectTo(Object o, String path) {
        try {
            ObjectOutputStream e = new ObjectOutputStream(newOutputStream(path));
            e.writeObject(o);
            e.close();
            return true;
        } catch (IOException var3) {
            Predefine.logger.warning("在保存对象" + o + "到" + path + "时发生异常" + var3);
            return false;
        }
    }

    public static Object readObjectFrom(String path) {
        ObjectInputStream ois = null;

        try {
            ois = new ObjectInputStream(newInputStream(path));
            Object e = ois.readObject();
            ois.close();
            return e;
        } catch (Exception var3) {
            Predefine.logger.warning("在从" + path + "读取对象时发生异常" + var3);
            return null;
        }
    }

    public static String readTxt(String path) {
        if(path == null) {
            return null;
        } else {
            try {
                Object e = Config.IOAdapter == null?new FileInputStream(path):Config.IOAdapter.open(path);
                byte[] fileContent = new byte[((InputStream)e).available()];
                readBytesFromOtherInputStream((InputStream)e, fileContent);
                ((InputStream)e).close();
                return new String(fileContent, Charset.forName("UTF-8"));
            } catch (FileNotFoundException var3) {
                Predefine.logger.warning("找不到" + path + var3);
                return null;
            } catch (IOException var4) {
                Predefine.logger.warning("读取" + path + "发生IO异常" + var4);
                return null;
            }
        }
    }

    public static LinkedList<String[]> readCsv(String path) {
        LinkedList resultList = new LinkedList();
        LinkedList lineList = readLineList(path);
        Iterator var3 = lineList.iterator();

        while(var3.hasNext()) {
            String line = (String)var3.next();
            resultList.add(line.split(","));
        }

        return resultList;
    }

    public static boolean saveTxt(String path, String content) {
        try {
            FileChannel e = (new FileOutputStream(path)).getChannel();
            e.write(ByteBuffer.wrap(content.getBytes()));
            e.close();
            return true;
        } catch (Exception var3) {
            Predefine.logger.throwing("IOUtil", "saveTxt", var3);
            Predefine.logger.warning("IOUtil saveTxt 到" + path + "失败" + var3.toString());
            return false;
        }
    }

    public static boolean saveTxt(String path, StringBuilder content) {
        return saveTxt(path, content.toString());
    }

    public static <T> boolean saveCollectionToTxt(Collection<T> collection, String path) {
        StringBuilder sb = new StringBuilder();
        Iterator var3 = collection.iterator();

        while(var3.hasNext()) {
            Object o = var3.next();
            sb.append(o);
            sb.append('\n');
        }

        return saveTxt(path, sb.toString());
    }

    public static byte[] readBytes(String path) {
        try {
            if(Config.IOAdapter == null) {
                return readBytesFromFileInputStream(new FileInputStream(path));
            } else {
                InputStream e = Config.IOAdapter.open(path);
                return e instanceof FileInputStream?readBytesFromFileInputStream((FileInputStream)e):readBytesFromOtherInputStream(e);
            }
        } catch (Exception var2) {
            Predefine.logger.warning("读取" + path + "时发生异常" + var2);
            return null;
        }
    }

    public static String readTxt(String file, String charsetName) throws IOException {
        InputStream is = Config.IOAdapter.open(file);
        byte[] targetArray = new byte[is.available()];

        int len;
        for(int off = 0; (len = is.read(targetArray, off, targetArray.length - off)) != -1 && off < targetArray.length; off += len) {
            ;
        }

        is.close();
        return new String(targetArray, charsetName);
    }

    public static String baseName(String path) {
        if(path != null && path.length() != 0) {
            path = path.replaceAll("[/\\\\]+", "/");
            int len = path.length();
            int upCount = 0;

            while(len > 0) {
                if(path.charAt(len - 1) == 47) {
                    --len;
                    if(len == 0) {
                        return "";
                    }
                }

                int lastInd = path.lastIndexOf(47, len - 1);
                String fileName = path.substring(lastInd + 1, len);
                if(fileName.equals(".")) {
                    --len;
                } else if(fileName.equals("src")) {
                    len -= 2;
                    ++upCount;
                } else {
                    if(upCount == 0) {
                        return fileName;
                    }

                    --upCount;
                    len -= fileName.length();
                }
            }

            return "";
        } else {
            return "";
        }
    }

    private static byte[] readBytesFromFileInputStream(FileInputStream fis) throws IOException {
        FileChannel channel = fis.getChannel();
        int fileSize = (int)channel.size();
        ByteBuffer byteBuffer = ByteBuffer.allocate(fileSize);
        channel.read(byteBuffer);
        byteBuffer.flip();
        byte[] bytes = byteBuffer.array();
        byteBuffer.clear();
        channel.close();
        fis.close();
        return bytes;
    }

    public static byte[] readBytesFromOtherInputStream(InputStream is) throws IOException {
        ByteArrayOutputStream data = new ByteArrayOutputStream();
        byte[] buffer = new byte[Math.max(is.available(), 4096)];

        int readBytes;
        while((readBytes = is.read(buffer, 0, buffer.length)) != -1) {
            data.write(buffer, 0, readBytes);
        }

        data.flush();
        return data.toByteArray();
    }

    public static int readBytesFromOtherInputStream(InputStream is, byte[] targetArray) throws IOException {
        assert targetArray != null;

        assert targetArray.length > 0;

        int len;
        int off;
        for(off = 0; off < targetArray.length && (len = is.read(targetArray, off, targetArray.length - off)) != -1; off += len) {
            ;
        }

        return off;
    }

    public static byte[] readBytesFromResource(String path) throws IOException {
        InputStream is = IOUtil.class.getResourceAsStream("/" + path);
        byte[] targetArray = new byte[is.available()];

        int len;
        for(int off = 0; (len = is.read(targetArray, off, targetArray.length - off)) != -1 && off < targetArray.length; off += len) {
            ;
        }

        is.close();
        return targetArray;
    }

    public static byte[] getBytes(InputStream is) throws IOException {
        short size = 1024;
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[size];

        int len;
        while((len = is.read(buf, 0, size)) != -1) {
            bos.write(buf, 0, len);
        }

        buf = bos.toByteArray();
        return buf;
    }

    public static LinkedList<String> readLineList(String path) {
        LinkedList result = new LinkedList();
        String txt = readTxt(path);
        if(txt == null) {
            return result;
        } else {
            StringTokenizer tokenizer = new StringTokenizer(txt, "\n");

            while(tokenizer.hasMoreTokens()) {
                result.add(tokenizer.nextToken());
            }

            return result;
        }
    }

    public static LinkedList<String> readLineListWithLessMemory(String path) {
        LinkedList result = new LinkedList();
        String line = null;

        try {
            BufferedReader e = new BufferedReader(new InputStreamReader(newInputStream(path), "UTF-8"));

            while((line = e.readLine()) != null) {
                result.add(line);
            }

            e.close();
        } catch (Exception var4) {
            Predefine.logger.warning("加载" + path + "失败，" + var4);
        }

        return result;
    }

    public static boolean saveMapToTxt(Map<Object, Object> map, String path) {
        return saveMapToTxt(map, path, "=");
    }

    public static boolean saveMapToTxt(Map<Object, Object> map, String path, String separator) {
        TreeMap map1 = new TreeMap(map);
        return saveEntrySetToTxt(map1.entrySet(), path, separator);
    }

    public static boolean saveEntrySetToTxt(Set<Entry<Object, Object>> entrySet, String path, String separator) {
        StringBuilder sbOut = new StringBuilder();
        Iterator var4 = entrySet.iterator();

        while(var4.hasNext()) {
            Entry entry = (Entry)var4.next();
            sbOut.append(entry.getKey());
            sbOut.append(separator);
            sbOut.append(entry.getValue());
            sbOut.append('\n');
        }

        return saveTxt(path, sbOut.toString());
    }

    public static String dirname(String path) {
        int index = path.lastIndexOf(47);
        return index == -1?path:path.substring(0, index + 1);
    }

    public static IOUtil.LineIterator readLine(String path) {
        return new IOUtil.LineIterator(path);
    }

    public static boolean isFileExists(String path) {
        return (new File(path)).exists();
    }

    public static boolean isResource(String path) {
        return path.startsWith("data/");
    }

    public static BufferedWriter newBufferedWriter(String path) throws IOException {
        return new BufferedWriter(new OutputStreamWriter(newOutputStream(path), "UTF-8"));
    }

    public static BufferedReader newBufferedReader(String path) throws IOException {
        return new BufferedReader(new InputStreamReader(newInputStream(path), "UTF-8"));
    }

    public static BufferedWriter newBufferedWriter(String path, boolean append) throws FileNotFoundException, UnsupportedEncodingException {
        return new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path, append), "UTF-8"));
    }

    public static InputStream newInputStream(String path) throws IOException {
        return (InputStream)(Config.IOAdapter == null?new FileInputStream(path):Config.IOAdapter.open(path));
    }

    public static OutputStream newOutputStream(String path) throws IOException {
        return (OutputStream)(Config.IOAdapter == null?new FileOutputStream(path):Config.IOAdapter.create(path));
    }

    public static String getSuffix(String name, String delimiter) {
        return name.substring(name.lastIndexOf(delimiter) + 1);
    }

    public static void writeLine(BufferedWriter bw, String... params) throws IOException {
        for(int i = 0; i < params.length - 1; ++i) {
            bw.write(params[i]);
            bw.write(9);
        }

        bw.write(params[params.length - 1]);
    }

    public static TreeMap<String, Attribute> loadDictionary(String... pathArray) throws IOException {
        TreeMap map = new TreeMap();
        String[] var2 = pathArray;
        int var3 = pathArray.length;

        for(int var4 = 0; var4 < var3; ++var4) {
            String path = var2[var4];
            BufferedReader br = new BufferedReader(new InputStreamReader(newInputStream(path), "UTF-8"));
            loadDictionary(br, map);
        }

        return map;
    }

    public static void loadDictionary(BufferedReader br, TreeMap<String, Attribute> storage) throws IOException {
        String line;
        while((line = br.readLine()) != null) {
            String[] param = line.split("\\s");
            int natureCount = (param.length - 1) / 2;
            Attribute attribute = new Attribute(natureCount);

            for(int i = 0; i < natureCount; ++i) {
                attribute.nature[i] = (Nature)Enum.valueOf(Nature.class, param[1 + 2 * i]);
                attribute.frequency[i] = Integer.parseInt(param[2 + 2 * i]);
                attribute.totalFrequency += attribute.frequency[i];
            }

            storage.put(param[0], attribute);
        }

        br.close();
    }

    public static void writeCustomNature(DataOutputStream out, LinkedHashSet<Nature> customNatureCollector) throws IOException {
        if(customNatureCollector.size() != 0) {
            out.writeInt(-customNatureCollector.size());
            Iterator var2 = customNatureCollector.iterator();

            while(var2.hasNext()) {
                Nature nature = (Nature)var2.next();
                TextUtility.writeString(nature.toString(), out);
            }

        }
    }

    public static InputStream getResourceAsStream(String path) throws FileNotFoundException {
        InputStream is = IOUtil.class.getResourceAsStream(path);
        if(is == null) {
            throw new FileNotFoundException("资源文件" + path + "不存在于jar中");
        } else {
            return is;
        }
    }

    public static class LineIterator implements Iterator<String> {
        BufferedReader bw;
        String line;

        public LineIterator(String path) {
            try {
                this.bw = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
                this.line = this.bw.readLine();
            } catch (FileNotFoundException var3) {
                Predefine.logger.warning("文件" + path + "不存在，接下来的调用会返回null\n" + TextUtility.exceptionToString(var3));
                this.bw = null;
            } catch (Exception var4) {
                Predefine.logger.warning("在读取过程中发生错误" + TextUtility.exceptionToString(var4));
                this.bw = null;
            }

        }

        public void close() {
            if(this.bw != null) {
                try {
                    this.bw.close();
                    this.bw = null;
                } catch (IOException var2) {
                    Predefine.logger.warning("关闭文件失败" + TextUtility.exceptionToString(var2));
                }

            }
        }

        public boolean hasNext() {
            if(this.bw == null) {
                return false;
            } else if(this.line == null) {
                try {
                    this.bw.close();
                    this.bw = null;
                } catch (IOException var2) {
                    Predefine.logger.warning("关闭文件失败" + TextUtility.exceptionToString(var2));
                }

                return false;
            } else {
                return true;
            }
        }

        public String next() {
            String preLine = this.line;

            try {
                if(this.bw != null) {
                    this.line = this.bw.readLine();
                    if(this.line == null && this.bw != null) {
                        try {
                            this.bw.close();
                            this.bw = null;
                        } catch (IOException var3) {
                            Predefine.logger.warning("关闭文件失败" + TextUtility.exceptionToString(var3));
                        }
                    }
                } else {
                    this.line = null;
                }
            } catch (IOException var4) {
                Predefine.logger.warning("在读取过程中发生错误" + TextUtility.exceptionToString(var4));
            }

            return preLine;
        }

        public void remove() {
            throw new UnsupportedOperationException("只读，不可写！");
        }
    }
}
