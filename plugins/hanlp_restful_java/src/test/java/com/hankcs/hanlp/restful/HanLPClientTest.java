package com.hankcs.hanlp.restful;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;

class HanLPClientTest
{
    HanLPClient client;

    @BeforeEach
    void setUp()
    {
        client = new HanLPClient("https://hanlp.hankcs.com/api", null);
    }

    @org.junit.jupiter.api.Test
    void parseText() throws IOException
    {
        Map<String, List> doc = client.parse("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。英首相与特朗普通电话讨论华为与苹果公司。");
        prettyPrint(doc);
    }

    @org.junit.jupiter.api.Test
    void parseSentences() throws IOException
    {
        Map<String, List> doc = client.parse(new String[]{
                "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。",
                "英首相与特朗普通电话讨论华为与苹果公司。"
        });
        prettyPrint(doc);
    }

    @org.junit.jupiter.api.Test
    void parseTokens() throws IOException
    {
        Map<String, List> doc = client.parse(new String[][]{
                new String[]{"2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多语种", "NLP", "技术", "。"},
                new String[]{"英", "首相", "与", "特朗普", "通", "电话", "讨论", "华为", "与", "苹果", "公司", "。"},
        });
        prettyPrint(doc);
    }

    @Test
    void parseCoarse() throws IOException
    {
        Map<String, List> doc = client.parse(
                "阿婆主来到北京立方庭参观自然语义科技公司。",
                new String[]{"tok/coarse", "pos", "dep"},
                new String[]{"tok/fine"});
        prettyPrint(doc);
    }

    @Test
    void textStyleTransfer() throws IOException
    {
        String doc = client.textStyleTransfer("国家对中石油抱有很大的期望.", "gov_doc");
        prettyPrint(doc);
    }

    void prettyPrint(Object object) throws JsonProcessingException
    {
        ObjectMapper mapper = new ObjectMapper();
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(object));
    }
}