package com.hankcs.hanlp.restful;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.hankcs.hanlp.restful.mrp.MeaningRepresentation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

class MeaningRepresentationTest
{


    @Test
    void parseText() throws IOException
    {
        String json = "[{\"id\": \"0\", \"input\": \"北京 大学 计算 语言学 研究所 和 富士通 研究 开发 中心 有限公司 ， 得到 了 人民日报社 新闻 信息 中心 的 语料库 。\", \"nodes\": [{\"id\": 0, \"label\": \"name\", \"properties\": [\"op1\", \"op2\"], \"values\": [\"北京\", \"大学\"], \"anchors\": [{\"from\": 0, \"to\": 2}, {\"from\": 3, \"to\": 5}]}, {\"id\": 1, \"label\": \"university\", \"anchors\": []}, {\"id\": 2, \"label\": \"name\", \"properties\": [\"op1\", \"op2\", \"op4\"], \"values\": [\"计算\", \"语言学\", \"\"], \"anchors\": [{\"from\": 6, \"to\": 8}, {\"from\": 9, \"to\": 12}, {\"from\": 13, \"to\": 16}]}, {\"id\": 3, \"label\": \"research-institute\", \"anchors\": []}, {\"id\": 4, \"label\": \"and\", \"anchors\": []}, {\"id\": 5, \"label\": \"name\", \"properties\": [\"op1\", \"op2\", \"op3\", \"op4\", \"op5\"], \"values\": [\"富士通\", \"研究\", \"开发\", \"中心\", \"有限公司\"], \"anchors\": [{\"from\": 19, \"to\": 22}, {\"from\": 23, \"to\": 25}, {\"from\": 26, \"to\": 28}, {\"from\": 29, \"to\": 31}, {\"from\": 32, \"to\": 36}]}, {\"id\": 6, \"label\": \"company\", \"anchors\": []}, {\"id\": 7, \"label\": \"得到-01\", \"anchors\": [{\"from\": 39, \"to\": 41}]}, {\"id\": 8, \"label\": \"了\", \"anchors\": [{\"from\": 42, \"to\": 43}]}, {\"id\": 9, \"label\": \"name\", \"properties\": [\"op1\"], \"values\": [\"人民日报社\"], \"anchors\": [{\"from\": 44, \"to\": 49}]}, {\"id\": 10, \"label\": \"organization\", \"anchors\": []}, {\"id\": 11, \"label\": \"name\", \"properties\": [\"op1\", \"op2\", \"op3\"], \"values\": [\"新闻\", \"信息\", \"中心\"], \"anchors\": [{\"from\": 50, \"to\": 52}, {\"from\": 53, \"to\": 55}, {\"from\": 56, \"to\": 58}]}, {\"id\": 12, \"label\": \"organization\", \"anchors\": []}, {\"id\": 13, \"label\": \"语料库\", \"anchors\": [{\"from\": 61, \"to\": 64}]}], \"edges\": [{\"source\": 7, \"target\": 8, \"label\": \"aspect\"}, {\"source\": 7, \"target\": 4, \"label\": \"arg0\"}, {\"source\": 10, \"target\": 9, \"label\": \"name\"}, {\"source\": 4, \"target\": 6, \"label\": \"op2\"}, {\"source\": 7, \"target\": 13, \"label\": \"arg1\"}, {\"source\": 6, \"target\": 5, \"label\": \"name\"}, {\"source\": 12, \"target\": 11, \"label\": \"name\"}, {\"source\": 3, \"target\": 2, \"label\": \"name\"}, {\"source\": 1, \"target\": 0, \"label\": \"name\"}, {\"source\": 13, \"target\": 12, \"label\": \"poss\"}, {\"source\": 4, \"target\": 3, \"label\": \"op1\"}, {\"source\": 12, \"target\": 9, \"label\": \"name\"}, {\"source\": 1, \"target\": 3, \"label\": \"part\"}], \"tops\": [7], \"framework\": \"amr\"}]";
        ObjectMapper mapper = new ObjectMapper();
        MeaningRepresentation[] graphs = mapper.readValue(json, MeaningRepresentation[].class);
        prettyPrint(graphs);
    }


    void prettyPrint(Object object) throws JsonProcessingException
    {
        ObjectMapper mapper = new ObjectMapper();
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(object));
    }
}