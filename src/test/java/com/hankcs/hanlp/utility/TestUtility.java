/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-23 11:05 PM</create-date>
 *
 * <copyright file="TestUtility.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.HanLP;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * @author hankcs
 */
public class TestUtility
{
    static
    {
        ensureFullData();
    }

    public static void ensureFullData()
    {
        ensureData(HanLP.Config.PerceptronCWSModelPath, "http://nlp.hankcs.com/download.php?file=data", HanLP.Config.PerceptronCWSModelPath.split("data")[0], false);
    }

    /**
     * 保证 name 存在，不存在时自动下载解压
     *
     * @param name 路径
     * @param url  下载地址
     * @return name的绝对路径
     */
    public static String ensureData(String name, String url)
    {
        return ensureData(name, url, null, true);
    }

    /**
     * 保证 name 存在，不存在时自动下载解压
     *
     * @param name 路径
     * @param url  下载地址
     * @return name的绝对路径
     */
    public static String ensureData(String name, String url, String parentPath, boolean overwrite)
    {
        File target = new File(name);
        if (target.exists()) return target.getAbsolutePath();
        try
        {
            File parentFile = parentPath == null ? new File(name).getParentFile() : new File(parentPath);
            if (!parentFile.exists()) parentFile.mkdirs();
            String filePath = downloadFile(url, parentFile.getAbsolutePath());
            if (filePath.endsWith(".zip"))
            {
                unzip(filePath, parentFile.getAbsolutePath(), overwrite);
            }
            return target.getAbsolutePath();
        }
        catch (Exception e)
        {
            System.err.printf("数据下载失败，请尝试手动下载 %s 到 %s 。原因如下：\n", url, target.getAbsolutePath());
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    /**
     * 保证 data/test/name 存在
     *
     * @param name
     * @param url
     * @return
     */
    public static String ensureTestData(String name, String url)
    {
        return ensureData(String.format("data/test/%s", name), url);
    }

    /**
     * Downloads a file from a URL
     *
     * @param fileURL  HTTP URL of the file to be downloaded
     * @param savePath path of the directory to save the file
     * @throws IOException
     * @author www.codejava.net
     */
    public static String downloadFile(String fileURL, String savePath)
        throws IOException
    {
        System.err.printf("Downloading %s to %s\n", fileURL, savePath);
        HttpURLConnection httpConn = request(fileURL);
        while (httpConn.getResponseCode() == HttpURLConnection.HTTP_MOVED_PERM || httpConn.getResponseCode() == HttpURLConnection.HTTP_MOVED_TEMP)
        {
            httpConn = request(httpConn.getHeaderField("Location"));
        }

        // always check HTTP response code first
        if (httpConn.getResponseCode() == HttpURLConnection.HTTP_OK)
        {
            String fileName = "";
            String disposition = httpConn.getHeaderField("Content-Disposition");
            String contentType = httpConn.getContentType();
            int contentLength = httpConn.getContentLength();

            if (disposition != null)
            {
                // extracts file name from header field
                int index = disposition.indexOf("filename=");
                if (index > 0)
                {
                    fileName = disposition.substring(index + 10,
                                                     disposition.length() - 1);
                }
            }
            else
            {
                // extracts file name from URL
                fileName = new File(httpConn.getURL().getPath()).getName();
            }

//            System.out.println("Content-Type = " + contentType);
//            System.out.println("Content-Disposition = " + disposition);
//            System.out.println("Content-Length = " + contentLength);
//            System.out.println("fileName = " + fileName);

            // opens input stream from the HTTP connection
            InputStream inputStream = httpConn.getInputStream();
            String saveFilePath = savePath;
            if (new File(savePath).isDirectory())
                saveFilePath = savePath + File.separator + fileName;
            String realPath;
            if (new File(saveFilePath).isFile())
            {
                System.err.printf("Use cached %s instead.\n", fileName);
                realPath = saveFilePath;
            }
            else
            {
                saveFilePath += ".downloading";

                // opens an output stream to save into file
                FileOutputStream outputStream = new FileOutputStream(saveFilePath);

                int bytesRead;
                byte[] buffer = new byte[4096];
                long start = System.currentTimeMillis();
                int progress_size = 0;
                while ((bytesRead = inputStream.read(buffer)) != -1)
                {
                    outputStream.write(buffer, 0, bytesRead);
                    long duration = (System.currentTimeMillis() - start) / 1000;
                    duration = Math.max(duration, 1);
                    progress_size += bytesRead;
                    int speed = (int) (progress_size / (1024 * duration));
                    float ratio = progress_size / (float) contentLength;
                    float percent = ratio * 100;
                    int eta = (int) (duration / ratio * (1 - ratio));
                    int minutes = eta / 60;
                    int seconds = eta % 60;

                    System.err.printf("\r%.2f%%, %d MB, %d KB/s, ETA %d min %d s", percent, progress_size / (1024 * 1024), speed, minutes, seconds);
                }
                System.err.println();
                outputStream.close();
                realPath = saveFilePath.substring(0, saveFilePath.length() - ".downloading".length());
                if (!new File(saveFilePath).renameTo(new File(realPath)))
                    throw new IOException("Failed to move file");
            }
            inputStream.close();
            httpConn.disconnect();

            return realPath;
        }
        else
        {
            httpConn.disconnect();
            throw new IOException("No file to download. Server replied HTTP code: " + httpConn.getResponseCode());
        }
    }

    private static HttpURLConnection request(String url) throws IOException
    {
        HttpURLConnection httpConn = (HttpURLConnection) new URL(url).openConnection();
        httpConn.setRequestProperty("User-Agent", "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.4; en-US; rv:1.9.2.2) Gecko/20100316 Firefox/3.6.2");
        return httpConn;
    }

    private static void unzip(String zipFilePath, String destDir, boolean overwrite)
    {
        System.err.println("Unzipping to " + destDir);
        File dir = new File(destDir);
        // create output directory if it doesn't exist
        if (!dir.exists()) dir.mkdirs();
        FileInputStream fis;
        //buffer for read and write data to file
        byte[] buffer = new byte[4096];
        try
        {
            fis = new FileInputStream(zipFilePath);
            ZipInputStream zis = new ZipInputStream(fis);
            ZipEntry ze = zis.getNextEntry();
            while (ze != null)
            {
                String fileName = ze.getName();
                File newFile = new File(destDir + File.separator + fileName);
                if (overwrite || !newFile.exists())
                {
                    if (ze.isDirectory())
                    {
                        //create directories for sub directories in zip
                        newFile.mkdirs();
                    }
                    else
                    {
                        new File(newFile.getParent()).mkdirs();
                        FileOutputStream fos = new FileOutputStream(newFile);
                        int len;
                        while ((len = zis.read(buffer)) > 0)
                        {
                            fos.write(buffer, 0, len);
                        }
                        fos.close();
                        //close this ZipEntry
                        zis.closeEntry();
                    }
                }
                ze = zis.getNextEntry();
            }
            //close last ZipEntry
            zis.closeEntry();
            zis.close();
            fis.close();
            new File(zipFilePath).delete();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
}
