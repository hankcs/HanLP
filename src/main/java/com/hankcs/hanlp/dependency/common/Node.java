/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 17:53</create-date>
 *
 * <copyright file="Node.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.common;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.corpus.dependency.CoNll.PosTagCompiler;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Term;

/**
 * 节点
 * @author hankcs
 */
public class Node
{
    public final static Node NULL = new Node(new Term(CoNLLWord.NULL.NAME, Nature.n), -1);
    static
    {
        NULL.label = "null";
    }
    public String word;
    public String compiledWord;
    public String label;
    public int id;

    public Node(Term term, int id)
    {
        this.id = id;
        switch (term.nature)
        {

            case bg:
                label = "b";
                break;
            case mg:
                label = "Mg";
                break;
            case nx:
                label = "x";
                break;
            case qg:
                label = "q";
                break;
            case ud:
                label = "u";
                break;
            case uj:
                label = "u";
                break;
            case uz:
                label = "uzhe";
                break;
            case ug:
                label = "uguo";
                break;
            case ul:
                label = "ulian";
                break;
            case uv:
                label = "u";
                break;
            case yg:
                label = "y";
                break;
            case zg:
                label = "z";
                break;
            case ntc:
            case ntcf:
            case ntcb:
            case ntch:
            case nto:
            case ntu:
            case nts:
            case nth:
                label = "nt";
                break;
            case nh:
            case nhm:
            case nhd:
                label = "nz";
                break;
            case nn:
                label = "n";
                break;
            case nnt:
                label = "n";
                break;
            case nnd:
                label = "n";
                break;
            case nf:
                label = "n";
                break;
            case ni:
            case nit:
            case nic:
                label = "nt";
                break;
            case nis:
                label = "n";
                break;
            case nm:
                label = "n";
                break;
            case nmc:
                label = "nz";
                break;
            case nb:
                label = "nz";
                break;
            case nba:
                label = "nz";
                break;
            case nbc:
            case nbp:
            case nz:
                label = "nz";
                break;
            case g:
                label = "nz";
                break;
            case gm:
            case gp:
            case gc:
            case gb:
            case gbc:
            case gg:
            case gi:
                label = "nz";
                break;
            case j:
                label = "nz";
                break;
            case i:
                label = "nz";
                break;
            case l:
                label = "nz";
                break;
            case rg:
            case Rg:
                label = "Rg";
                break;
            case udh:
                label = "u";
                break;
            case e:
                label = "y";
                break;
            case xx:
                label = "x";
                break;
            case xu:
                label = "x";
                break;
            case w:
            case wkz:
            case wky:
            case wyz:
            case wyy:
            case wj:
            case ww:
            case wt:
            case wd:
            case wf:
            case wn:
            case wm:
            case ws:
            case wp:
            case wb:
            case wh:
                label = "x";
                break;
            case begin:
                label = "root";
                break;
            default:
                label = term.nature.toString();
                break;
        }
        word = term.word;
        compiledWord = PosTagCompiler.compile(label, word);
    }

    @Override
    public String toString()
    {
        return word + "/" + label;
    }
}
