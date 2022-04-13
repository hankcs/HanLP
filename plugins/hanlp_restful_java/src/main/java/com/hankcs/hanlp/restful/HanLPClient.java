/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2020-12-26 11:54 PM</create-date>
 *
 * <copyright file="HanLPClient.java">
 * Copyright (c) 2020, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.restful;


import com.fasterxml.jackson.databind.ObjectMapper;
import com.hankcs.hanlp.restful.mrp.MeaningRepresentation;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * A RESTful client implementing the data format specification of HanLP.
 *
 * @author hankcs
 * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
 */
public class HanLPClient
{
    private String url;
    private String auth;
    private String language;
    private int timeout;
    private ObjectMapper mapper;

    /**
     * @param url      An API endpoint to a service provider.
     * @param auth     An auth key licenced by a service provider.
     * @param language The language this client will be expecting. Contact the service provider for the list of
     *                 languages supported. Conventionally, zh is used for Chinese and mul for multilingual.
     *                 Leave null to use the default language on server.
     * @param timeout  Maximum waiting time in seconds for a request.
     */
    public HanLPClient(String url, String auth, String language, int timeout)
    {
        if (auth == null)
        {
            auth = System.getenv().getOrDefault("HANLP_AUTH", null);
        }
        this.url = url;
        this.auth = auth;
        this.language = language;
        this.timeout = timeout * 1000;
        this.mapper = new ObjectMapper();
    }

    /**
     * @param url  An API endpoint to a service provider.
     * @param auth An auth key licenced by a service provider.
     */
    public HanLPClient(String url, String auth)
    {
        this(url, auth, null, 5);
    }

    /**
     * Parse a raw document.
     *
     * @param text      Document content which can have multiple sentences.
     * @param tasks     Tasks to perform.
     * @param skipTasks Tasks to skip.
     * @return Parsed annotations.
     * @throws IOException HTTP exception.
     * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
     */
    public Map<String, List> parse(String text, String[] tasks, String[] skipTasks) throws IOException
    {
        //noinspection unchecked
        return mapper.readValue(post("/parse", new DocumentInput(text, tasks, skipTasks, language)), Map.class);
    }

    /**
     * Parse a raw document.
     *
     * @param text Document content which can have multiple sentences.
     * @return Parsed annotations.
     * @throws IOException HTTP exception.
     * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
     */
    public Map<String, List> parse(String text) throws IOException
    {
        return parse(text, null, null);
    }

    /**
     * Parse an array of sentences.
     *
     * @param sentences Multiple sentences to parse.
     * @param tasks     Tasks to perform.
     * @param skipTasks Tasks to skip.
     * @return Parsed annotations.
     * @throws IOException HTTP exception.
     * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
     */
    public Map<String, List> parse(String[] sentences, String[] tasks, String[] skipTasks) throws IOException
    {
        //noinspection unchecked
        return mapper.readValue(post("/parse", new SentenceInput(sentences, tasks, skipTasks, language)), Map.class);
    }

    /**
     * Parse an array of sentences.
     *
     * @param sentences Multiple sentences to parse.
     * @return Parsed annotations.
     * @throws IOException HTTP exception.
     * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
     */
    public Map<String, List> parse(String[] sentences) throws IOException
    {
        return parse(sentences, null, null);
    }

    /**
     * Parse an array of pre-tokenized sentences.
     *
     * @param tokens    Multiple pre-tokenized sentences to parse.
     * @param tasks     Tasks to perform.
     * @param skipTasks Tasks to skip.
     * @return Parsed annotations.
     * @throws IOException HTTP exception.
     * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
     */
    public Map<String, List> parse(String[][] tokens, String[] tasks, String[] skipTasks) throws IOException
    {
        //noinspection unchecked
        return mapper.readValue(post("/parse", new TokenInput(tokens, tasks, skipTasks, language)), Map.class);
    }

    /**
     * Parse an array of pre-tokenized sentences.
     *
     * @param tokens Multiple pre-tokenized sentences to parse.
     * @return Parsed annotations.
     * @throws IOException HTTP exception.
     * @see <a href="https://hanlp.hankcs.com/docs/data_format.html">Data Format</a>
     */
    public Map<String, List> parse(String[][] tokens) throws IOException
    {
        return parse(tokens, null, null);
    }

    /**
     * Split a document into sentences and tokenize them.
     *
     * @param text   A document.
     * @param coarse Whether to perform coarse-grained or fine-grained tokenization.
     * @return A list of tokenized sentences.
     * @throws IOException HTTP exception.
     */
    public List<List<String>> tokenize(String text, Boolean coarse) throws IOException
    {
        String[] tasks;
        if (coarse != null)
        {
            if (coarse)
                tasks = new String[]{"tok/coarse"};
            else
                tasks = new String[]{"tok/fine"};
        }
        else
            tasks = new String[]{"tok"};
        Map<String, List> doc = parse(text, tasks, null);
        //noinspection unchecked
        return doc.values().iterator().next();
    }

    /**
     * Split a document into sentences and tokenize them using fine-grained standard.
     *
     * @param text A document.
     * @return A list of tokenized sentences.
     * @throws IOException HTTP exception.
     */
    public List<List<String>> tokenize(String text) throws IOException
    {
        return tokenize(text, null);
    }

    /**
     * Text style transfer aims to change the style of the input text to the target style while preserving its content.
     *
     * @param text        Source text.
     * @param targetStyle Target style.
     * @return Text of the target style.
     */
    public List<String> textStyleTransfer(List<String> text, String targetStyle) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("text", text);
        input.put("target_style", targetStyle);
        input.put("language", language);
        //noinspection unchecked
        return mapper.readValue(post("/text_style_transfer", input), List.class);
    }

    /**
     * Text style transfer aims to change the style of the input text to the target style while preserving its content.
     *
     * @param text        Source text.
     * @param targetStyle Target style.
     * @return Text of the target style.
     */
    public String textStyleTransfer(String text, String targetStyle) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("text", text);
        input.put("target_style", targetStyle);
        input.put("language", language);
        return mapper.readValue(post("/text_style_transfer", input), String.class);
    }

    /**
     * Semantic textual similarity deals with determining how similar two pieces of texts are.
     *
     * @param textA The first text.
     * @param textB The second text.
     * @return Their similarity.
     * @throws IOException HTTP errors.
     */
    public Float semanticTextualSimilarity(String textA, String textB) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("text", new String[]{textA, textB});
        input.put("language", language);
        return mapper.readValue(post("/semantic_textual_similarity", input), Float.class);
    }

    /**
     * Semantic textual similarity deals with determining how similar two pieces of texts are.
     *
     * @param text The pairs of text.
     * @return Their similarities.
     * @throws IOException HTTP errors.
     */
    public List<Float> semanticTextualSimilarity(String[][] text) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("text", text);
        input.put("language", language);
        //noinspection unchecked
        return mapper.readValue(post("/semantic_textual_similarity", input), List.class);
    }

    /**
     * Coreference resolution is the task of clustering mentions in text that refer to the same underlying real world entities.
     *
     * @param text A piece of text, usually a document without tokenization.
     * @return Coreference resolution clusters and tokens.
     * @throws IOException HTTP errors.
     */
    public CoreferenceResolutionOutput coreferenceResolution(String text) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("text", text);
        input.put("language", language);
        //noinspection unchecked
        Map<String, List> response = mapper.readValue(post("/coreference_resolution", input), Map.class);
        //noinspection unchecked
        List<List<List>> clusters = response.get("clusters");
        return new CoreferenceResolutionOutput(_convert_clusters(clusters), (ArrayList<String>) response.get("tokens"));
    }

    /**
     * Coreference resolution is the task of clustering mentions in text that refer to the same underlying real world entities.
     *
     * @param tokens   A list of sentences where each sentence is a list of tokens.
     * @param speakers A list of speakers where each speaker is a String representing the speaker's ID, e.g., "Tom".
     * @return Coreference resolution clusters.
     * @throws IOException HTTP errors.
     */
    public List<Set<Span>> coreferenceResolution(String[][] tokens, String[] speakers) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("tokens", tokens);
        input.put("speakers", speakers);
        input.put("language", language);
        //noinspection unchecked
        List<List<List>> clusters = mapper.readValue(post("/coreference_resolution", input), List.class);
        return _convert_clusters(clusters);
    }

    /**
     * Coreference resolution is the task of clustering mentions in text that refer to the same underlying real world entities.
     *
     * @param tokens A list of sentences where each sentence is a list of tokens.
     * @return Coreference resolution clusters.
     * @throws IOException HTTP errors.
     */
    public List<Set<Span>> coreferenceResolution(String[][] tokens) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("tokens", tokens);
        input.put("language", language);
        //noinspection unchecked
        List<List<List>> clusters = mapper.readValue(post("/coreference_resolution", input), List.class);
        return _convert_clusters(clusters);
    }

    private static List<Set<Span>> _convert_clusters(List<List<List>> clusters)
    {
        List<Set<Span>> results = new ArrayList<>(clusters.size());
        for (List<List> cluster : clusters)
        {
            Set<Span> spans = new LinkedHashSet<>();
            for (List span : cluster)
            {
                spans.add(new Span((String) span.get(0), (Integer) span.get(1), (Integer) span.get(2)));
            }
            results.add(spans);
        }
        return results;
    }

    /**
     * Abstract Meaning Representation (AMR) captures “who is doing what to whom” in a sentence. Each sentence is
     * represented as a rooted, directed, acyclic graph consisting of nodes (concepts) and edges (relations).
     *
     * @param text A piece of text, usually a document without tokenization.
     * @return AMR graphs.
     * @throws IOException HTTP errors.
     */
    public MeaningRepresentation[] abstractMeaningRepresentation(String text) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("text", text);
        input.put("language", language);
        return mapper.readValue(post("/abstract_meaning_representation", input), MeaningRepresentation[].class);
    }

    /**
     * Abstract Meaning Representation (AMR) captures “who is doing what to whom” in a sentence. Each sentence is
     * represented as a rooted, directed, acyclic graph consisting of nodes (concepts) and edges (relations).
     *
     * @param tokens A list of sentences where each sentence is a list of tokens.
     * @return AMR graphs.
     * @throws IOException HTTP errors.
     */
    public MeaningRepresentation[] abstractMeaningRepresentation(String[][] tokens) throws IOException
    {
        Map<String, Object> input = new HashMap<>();
        input.put("tokens", tokens);
        input.put("language", language);
        return mapper.readValue(post("/abstract_meaning_representation", input), MeaningRepresentation[].class);
    }

    private String post(String api, Object input_) throws IOException
    {
        URL url = new URL(this.url + api);

        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        if (auth != null)
            con.setRequestProperty("Authorization", "Basic " + auth);
        con.setRequestProperty("Content-Type", "application/json; utf-8");
        con.setRequestProperty("Accept", "application/json");
        con.setDoOutput(true);
        con.setConnectTimeout(timeout);
        con.setReadTimeout(timeout);

        String jsonInputString = mapper.writeValueAsString(input_);

        try (OutputStream os = con.getOutputStream())
        {
            byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        int code = con.getResponseCode();
        if (code != 200)
        {
            StringBuilder response = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(con.getErrorStream(), StandardCharsets.UTF_8)))
            {
                String responseLine;
                while ((responseLine = br.readLine()) != null)
                {
                    response.append(responseLine.trim());
                }
            }
            String error = String.format("Request failed, status code = %d, error = %s", code, con.getResponseMessage());
            try
            {
                Map detail = mapper.readValue(response.toString(), Map.class);
                error = (String) detail.get("detail");
            }
            catch (Exception ignored)
            {
            }
            throw new IOException(error);
        }

        StringBuilder response = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8)))
        {
            String responseLine;
            while ((responseLine = br.readLine()) != null)
            {
                response.append(responseLine.trim());
            }
        }
        return response.toString();
    }

}
