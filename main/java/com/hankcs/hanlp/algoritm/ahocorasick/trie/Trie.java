package com.hankcs.hanlp.algoritm.ahocorasick.trie;

import com.hankcs.hanlp.algoritm.ahocorasick.interval.IntervalTree;
import com.hankcs.hanlp.algoritm.ahocorasick.interval.Intervalable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * 基于 Aho-Corasick 白皮书, 贝尔实验室: ftp://163.13.200.222/assistant/bearhero/prog/%A8%E4%A5%A6/ac_bm.pdf
 *
 * @author Robert Bor
 */
public class Trie
{

    private TrieConfig trieConfig;

    private State rootState;

    /**
     * 是否建立了failure表
     */
    private boolean failureStatesConstructed = false;

    /**
     * 构造一棵trie树
     * @param trieConfig
     */
    public Trie(TrieConfig trieConfig)
    {
        this.trieConfig = trieConfig;
        this.rootState = new State();
    }

    /**
     * 以默认配置构造一棵trie树
     */
    public Trie()
    {
        this(new TrieConfig());
    }

    /**
     * 大小写敏感
     * @return
     */
    public Trie caseInsensitive()
    {
        this.trieConfig.setCaseInsensitive(true);
        return this;
    }

    /**
     * 不允许模式串在位置上前后重叠
     * @return
     */
    public Trie removeOverlaps()
    {
        this.trieConfig.setAllowOverlaps(false);
        return this;
    }

    /**
     * 只匹配完整单词
     * @return
     */
    public Trie onlyWholeWords()
    {
        this.trieConfig.setOnlyWholeWords(true);
        return this;
    }

    /**
     * 添加一个模式串
     * @param keyword
     */
    public void addKeyword(String keyword)
    {
        if (keyword == null || keyword.length() == 0)
        {
            return;
        }
        State currentState = this.rootState;
        for (Character character : keyword.toCharArray())
        {
            currentState = currentState.addState(character);
        }
        currentState.addEmit(keyword);
    }

    /**
     * 一个分词器
     * @param text 待分词文本
     * @return
     */
    public Collection<Token> tokenize(String text)
    {

        Collection<Token> tokens = new ArrayList<Token>();

        Collection<Emit> collectedEmits = parseText(text);
        int lastCollectedPosition = -1;
        for (Emit emit : collectedEmits)
        {
            if (emit.getStart() - lastCollectedPosition > 1)
            {
                tokens.add(createFragment(emit, text, lastCollectedPosition));
            }
            tokens.add(createMatch(emit, text));
            lastCollectedPosition = emit.getEnd();
        }
        if (text.length() - lastCollectedPosition > 1)
        {
            tokens.add(createFragment(null, text, lastCollectedPosition));
        }

        return tokens;
    }

    private Token createFragment(Emit emit, String text, int lastCollectedPosition)
    {
        return new FragmentToken(text.substring(lastCollectedPosition + 1, emit == null ? text.length() : emit.getStart()));
    }

    private Token createMatch(Emit emit, String text)
    {
        return new MatchToken(text.substring(emit.getStart(), emit.getEnd() + 1), emit);
    }

    /**
     * 模式匹配
     * @param text 待匹配的文本
     * @return 匹配到的模式串
     */
    @SuppressWarnings("unchecked")
    public Collection<Emit> parseText(String text)
    {
        checkForConstructedFailureStates();

        int position = 0;
        State currentState = this.rootState;
        List<Emit> collectedEmits = new ArrayList<Emit>();
        for (Character character : text.toCharArray())
        {
            if (trieConfig.isCaseInsensitive())
            {
                character = Character.toLowerCase(character);
            }
            currentState = getState(currentState, character);
            storeEmits(position, currentState, collectedEmits);
            ++position;
        }

        if (trieConfig.isOnlyWholeWords())
        {
            removePartialMatches(text, collectedEmits);
        }

        if (!trieConfig.isAllowOverlaps())
        {
            IntervalTree intervalTree = new IntervalTree((List<Intervalable>) (List<?>) collectedEmits);
            intervalTree.removeOverlaps((List<Intervalable>) (List<?>) collectedEmits);
        }

        return collectedEmits;
    }

    /**
     * 移除半截单词
     * @param searchText
     * @param collectedEmits
     */
    private void removePartialMatches(String searchText, List<Emit> collectedEmits)
    {
        long size = searchText.length();
        List<Emit> removeEmits = new ArrayList<Emit>();
        for (Emit emit : collectedEmits)
        {
            if ((emit.getStart() == 0 ||
                    !Character.isAlphabetic(searchText.charAt(emit.getStart() - 1))) &&
                    (emit.getEnd() + 1 == size ||
                            !Character.isAlphabetic(searchText.charAt(emit.getEnd() + 1))))
            {
                continue;
            }
            removeEmits.add(emit);
        }

        for (Emit removeEmit : removeEmits)
        {
            collectedEmits.remove(removeEmit);
        }
    }

    /**
     * 跳转到下一个状态
     * @param currentState 当前状态
     * @param character 接受字符
     * @return 跳转结果
     */
    private static State getState(State currentState, Character character)
    {
        State newCurrentState = currentState.nextState(character);  // 先按success跳转
        while (newCurrentState == null) // 跳转失败的话，按failure跳转
        {
            currentState = currentState.failure();
            newCurrentState = currentState.nextState(character);
        }
        return newCurrentState;
    }

    /**
     * 检查是否建立了failure表
     */
    private void checkForConstructedFailureStates()
    {
        if (!this.failureStatesConstructed)
        {
            constructFailureStates();
        }
    }

    /**
     * 建立failure表
     */
    private void constructFailureStates()
    {
        Queue<State> queue = new LinkedBlockingDeque<State>();

        // 第一步，将深度为1的节点的failure设为根节点
        for (State depthOneState : this.rootState.getStates())
        {
            depthOneState.setFailure(this.rootState);
            queue.add(depthOneState);
        }
        this.failureStatesConstructed = true;

        // 第二步，为深度 > 1 的节点建立failure表，这是一个bfs
        while (!queue.isEmpty())
        {
            State currentState = queue.remove();

            for (Character transition : currentState.getTransitions())
            {
                State targetState = currentState.nextState(transition);
                queue.add(targetState);

                State traceFailureState = currentState.failure();
                while (traceFailureState.nextState(transition) == null)
                {
                    traceFailureState = traceFailureState.failure();
                }
                State newFailureState = traceFailureState.nextState(transition);
                targetState.setFailure(newFailureState);
                targetState.addEmit(newFailureState.emit());
            }
        }
    }

    /**
     * 保存匹配结果
     * @param position 当前位置，也就是匹配到的模式串的结束位置+1
     * @param currentState 当前状态
     * @param collectedEmits 保存位置
     */
    private static void storeEmits(int position, State currentState, List<Emit> collectedEmits)
    {
        Collection<String> emits = currentState.emit();
        if (emits != null && !emits.isEmpty())
        {
            for (String emit : emits)
            {
                collectedEmits.add(new Emit(position - emit.length() + 1, position, emit));
            }
        }
    }

}
