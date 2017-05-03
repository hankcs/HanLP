package com.hankcs.test;



import org.junit.Test;

import java.io.File;

/**
 * Created by linming on 2017/3/7.
 */
public class CleanFile {

    String PATH = "C:/code/github/HanLP-hankcs";

    @Test
    public void cleanBinUnderDicts() {
        String path = PATH + "/dicts/";
        deleteFile(path, ".txt.bin");
    }

    @Test
    public void cleanBinUnderHanlpCustom() {
        String path = PATH + "/data/dictionary/custom";
        deleteFile(path, ".txt.bin");
    }

    @Test
    public void cleanBinUnderHanlpCore() {
        String path = PATH;

        String file1 = path + "/data/dictionary/stopwords.txt.bin";
        deleteFile(file1, ".bin");

        String file2 = path + "/data/dictionary/CoreNatureDictionary.txt.bin";
        deleteFile(file2, ".bin");

        String file3 = path + "/data/dictionary/CoreNatureDictionary.tr.txt.bin";
        deleteFile(file3, ".bin");

        String file4 = path + "/data/dictionary/CoreNatureDictionary.ngram.txt.table.bin";
        deleteFile(file4, ".bin");

        String file5 = path + "/data/dictionary/CoreNatureDictionary.ngram.mini.txt.table.bin";
        deleteFile(file5, ".bin");

        String file6 = path + "/data/dictionary/CoreNatureDictionary.mini.txt.bin";
        deleteFile(file6, ".bin");
    }

    @Test
    public void cleanBinUnderHanlpDat() {
        String path = PATH + "/data/dictionary/";
        deleteFile(path, ".dat");
    }


    @Test
    public void cleanAll() {
        cleanBinUnderDicts();
        cleanBinUnderHanlpCustom();
        cleanBinUnderHanlpCore();
        cleanBinUnderHanlpDat();
    }

    public static void deleteFile(String strPath, String suffix) {
        File dir = new File(strPath);
        if(dir != null && dir.isFile() && dir.getName().endsWith(suffix)) {
            String strFileName = dir.getAbsolutePath();
            dir.delete();
            System.out.println("---" + strFileName);
        }

        File[] files = dir.listFiles(); // 该文件目录下文件全部放入数组
        if (files != null) {
            for (int i = 0; i < files.length; i++) {
                deleteFile(files[i].getAbsolutePath(), suffix);
            }

        }
    }

}
