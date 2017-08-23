package com.hankcs.hanlp.collection.trie;

import java.io.Serializable;

/**
 * Created by chenjianfeng on 2017/7/26.
 */
public class Node implements Serializable{
    int code;
    int depth;
    int left;
    int right;

    @Override
    public String toString()
    {
        return "Node{" +
            "code=" + code +
            ", depth=" + depth +
            ", left=" + left +
            ", right=" + right +
            '}';
    }
}
