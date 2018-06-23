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
    public static String ensureData(String name, String url)
    {
        File target = new File(String.format("data/test/%s", name));
        if (target.exists()) return target.getAbsolutePath();
        try
        {
            File testPath = new File("data/test");
            if (!testPath.exists()) testPath.mkdirs();
            String filePath = downloadFile(url, testPath.getAbsolutePath());
            if (filePath.endsWith(".zip"))
            {
                System.err.println("Unzipping to " + target.getAbsolutePath());
                unzip(filePath, testPath.getAbsolutePath());
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
        URL url = new URL(fileURL);
        HttpURLConnection httpConn = (HttpURLConnection) url.openConnection();
        int responseCode = httpConn.getResponseCode();

        // always check HTTP response code first
        if (responseCode == HttpURLConnection.HTTP_OK)
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
                fileName = fileURL.substring(fileURL.lastIndexOf("/") + 1,
                                             fileURL.length());
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

            outputStream.close();
            inputStream.close();
            httpConn.disconnect();
            String realPath = saveFilePath.substring(0, saveFilePath.length() - ".downloading".length());
            if (!new File(saveFilePath).renameTo(new File(realPath)))
                throw new IOException("Failed to move file");
            System.err.println();
            return realPath;
        }
        else
        {
            httpConn.disconnect();
            throw new IOException("No file to download. Server replied HTTP code: " + responseCode);
        }
    }

    private static void unzip(String zipFilePath, String destDir)
    {
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
                //create directories for sub directories in zip
                if (ze.isDirectory())
                {
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
                ze = zis.getNextEntry();
            }
            //close last ZipEntry
            zis.closeEntry();
            zis.close();
            fis.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

    }
}
