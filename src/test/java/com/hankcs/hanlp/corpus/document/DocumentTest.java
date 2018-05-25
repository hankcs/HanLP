package com.hankcs.hanlp.corpus.document;

import junit.framework.TestCase;

public class DocumentTest extends TestCase
{
    public void testCreate() throws Exception
    {
        Document document = Document.create("[上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt 董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国/ns 纽约/ns 现代/t 艺术/n 博物馆/n]/ns 参观/v");
        assertNotNull(document);
    }
}