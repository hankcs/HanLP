
package com.hankcs.hanlp.corpus.io;

import java.io.*;

public class ResourceIOAdapter implements IIOAdapter {
    public ResourceIOAdapter() {
    }

    public InputStream open(String path) throws IOException {
        return (InputStream)(IOUtil.isResource(path)?IOUtil.getResourceAsStream("/" + path):new FileInputStream(path));
    }

    public OutputStream create(String path) throws IOException {
        if(IOUtil.isResource(path)) {
            throw new IllegalArgumentException("不支持写入jar包资源路径" + path);
        } else {
            return new FileOutputStream(path);
        }
    }
}
