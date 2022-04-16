package com.hankcs.hanlp.restful;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
    void tokenize() throws IOException
    {
        List<List<String>> fine = client.tokenize("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。");
        System.out.println(fine);
        List<List<String>> coarse = client.tokenize("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。", true);
        System.out.println(coarse);
    }

    @Test
    void textStyleTransfer() throws IOException
    {
        String doc = client.textStyleTransfer("国家对中石油抱有很大的期望.", "gov_doc");
        prettyPrint(doc);
    }

    @Test
    void semanticTextualSimilarity() throws IOException
    {
        Float similarity = client.semanticTextualSimilarity("看图猜一电影名", "看图猜电影");
        prettyPrint(similarity);
    }

    @Test
    void coreferenceResolutionText() throws IOException
    {
        CoreferenceResolutionOutput clusters = client.coreferenceResolution("我姐送我她的猫。我很喜欢它。");
        prettyPrint(clusters);
    }

    @Test
    void coreferenceResolutionTokens() throws IOException
    {
        List<Set<Span>> clusters = client.coreferenceResolution(
                new String[][]{
                        new String[]{"我", "姐", "送", "我", "她", "的", "猫", "。"},
                        new String[]{"我", "很", "喜欢", "它", "。"}});
        prettyPrint(clusters);
    }

    @Test
    void coreferenceResolutionTokensWithSpeakers() throws IOException
    {
        List<Set<Span>> clusters = client.coreferenceResolution(
                new String[][]{
                        new String[]{"我", "姐", "送", "我", "她", "的", "猫", "。"},
                        new String[]{"我", "很", "喜欢", "它", "。"}},
                new String[]{"张三", "张三"});
        prettyPrint(clusters);
    }

    @Test
    void keyphraseExtraction() throws IOException
    {
        prettyPrint(client.keyphraseExtraction(
                "自然语言处理是一门博大精深的学科，掌握理论才能发挥出HanLP的全部性能。" +
                        "《自然语言处理入门》是一本配套HanLP的NLP入门书，助你零起点上手自然语言处理。", 3));
    }

    @Test
    void abstractMeaningRepresentationText() throws IOException
    {
        prettyPrint(client.abstractMeaningRepresentation("男孩希望女孩相信他。阿婆主来到北京立方庭参观自然语义科技公司。"));
    }

    @Test
    void abstractMeaningRepresentationTokens() throws IOException
    {
        prettyPrint(client.abstractMeaningRepresentation(new String[][]{
                new String[]{"2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多语种", "NLP", "技术", "。"},
                new String[]{"英", "首相", "与", "特朗普", "通", "电话", "讨论", "华为", "与", "苹果", "公司", "。"}}));
    }

    void prettyPrint(Object object) throws JsonProcessingException
    {
        ObjectMapper mapper = new ObjectMapper();
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(object));
    }
}