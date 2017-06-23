package com.hankcs.hanlp.collection.AhoCorasick;

import java.util.*;

/**
 * <p>
 * 一个状态有如下几个功能
 * </p>
 * <p/>
 * <ul>
 * <li>success; 成功转移到另一个状态</li>
 * <li>failure; 不可顺着字符串跳转的话，则跳转到一个浅一点的节点</li>
 * <li>emits; 命中一个模式串</li>
 * </ul>
 * <p/>
 * <p>
 * 根节点稍有不同，根节点没有 failure 功能，它的“failure”指的是按照字符串路径转移到下一个状态。其他节点则都有failure状态。
 * </p>
 *
 * @author Robert Bor
 */
public class State
{

    /**
     * 模式串的长度，也是这个状态的深度
     */
    protected final int depth;

    /**
     * fail 函数，如果没有匹配到，则跳转到此状态。
     */
    private State failure = null;

    /**
     * 只要这个状态可达，则记录模式串
     */
    private Set<Integer> emits = null;
    /**
     * goto 表，也称转移函数。根据字符串的下一个字符转移到下一个状态
     */
    private Map<Character, State> success = new TreeMap<Character, State>();

    /**
     * 在双数组中的对应下标
     */
    private int index;

    /**
     * 构造深度为0的节点
     */
    public State()
    {
        this(0);
    }

    /**
     * 构造深度为depth的节点
     * @param depth
     */
    public State(int depth)
    {
        this.depth = depth;
    }

    /**
     * 获取节点深度
     * @return
     */
    public int getDepth()
    {
        return this.depth;
    }

    /**
     * 添加一个匹配到的模式串（这个状态对应着这个模式串)
     * @param keyword
     */
    public void addEmit(int keyword)
    {
        if (this.emits == null)
        {
            this.emits = new TreeSet<Integer>(Collections.reverseOrder());
        }
        this.emits.add(keyword);
    }

    /**
     * 获取最大的值
     * @return
     */
    public Integer getLargestValueId()
    {
        if (emits == null || emits.size() == 0) return null;

        return emits.iterator().next();
    }

    /**
     * 添加一些匹配到的模式串
     * @param emits
     */
    public void addEmit(Collection<Integer> emits)
    {
        for (int emit : emits)
        {
            addEmit(emit);
        }
    }

    /**
     * 获取这个节点代表的模式串（们）
     * @return
     */
    public Collection<Integer> emit()
    {
        return this.emits == null ? Collections.<Integer>emptyList() : this.emits;
    }

    /**
     * 是否是终止状态
     * @return
     */
    public boolean isAcceptable()
    {
        return this.depth > 0 && this.emits != null;
    }

    /**
     * 获取failure状态
     * @return
     */
    public State failure()
    {
        return this.failure;
    }

    /**
     * 设置failure状态
     * @param failState
     */
    public void setFailure(State failState, int fail[])
    {
        this.failure = failState;
        fail[index] = failState.index;
    }

    /**
     * 转移到下一个状态
     * @param character 希望按此字符转移
     * @param ignoreRootState 是否忽略根节点，如果是根节点自己调用则应该是true，否则为false
     * @return 转移结果
     */
    private State nextState(Character character, boolean ignoreRootState)
    {
        State nextState = this.success.get(character);
        if (!ignoreRootState && nextState == null && this.depth == 0)
        {
            nextState = this;
        }
        return nextState;
    }

    /**
     * 按照character转移，根节点转移失败会返回自己（永远不会返回null）
     * @param character
     * @return
     */
    public State nextState(Character character)
    {
        return nextState(character, false);
    }

    /**
     * 按照character转移，任何节点转移失败会返回null
     * @param character
     * @return
     */
    public State nextStateIgnoreRootState(Character character)
    {
        return nextState(character, true);
    }

    public State addState(Character character)
    {
        State nextState = nextStateIgnoreRootState(character);
        if (nextState == null)
        {
            nextState = new State(this.depth + 1);
            this.success.put(character, nextState);
        }
        return nextState;
    }

    public Collection<State> getStates()
    {
        return this.success.values();
    }

    public Collection<Character> getTransitions()
    {
        return this.success.keySet();
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder("State{");
        sb.append("depth=").append(depth);
        sb.append(", ID=").append(index);
        sb.append(", emits=").append(emits);
        sb.append(", success=").append(success.keySet());
        sb.append(", failureID=").append(failure == null ? "-1" : failure.index);
        sb.append(", failure=").append(failure);
        sb.append('}');
        return sb.toString();
    }

    /**
     * 获取goto表
     * @return
     */
    public Map<Character, State> getSuccess()
    {
        return success;
    }

    public int getIndex()
    {
        return index;
    }

    public void setIndex(int index)
    {
        this.index = index;
    }
}
