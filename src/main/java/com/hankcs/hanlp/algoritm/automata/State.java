package com.hankcs.hanlp.algoritm.automata;

import java.util.*;

public class State
{

    /**
     * 模式串的长度，也是这个状态的深度
     */
    protected final int depth;

    /**
     * 只要这个状态可达，则记录模式串
     */
    private Integer emits = null;
    /**
     * goto 表，也称转移函数。根据字符串的下一个字符转移到下一个状态
     */
    private Map<Integer, State> success = new TreeMap<Integer, State>();

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
     *
     * @param depth
     */
    public State(int depth)
    {
        this.depth = depth;
    }

    /**
     * 获取节点深度
     *
     * @return
     */
    public int getDepth()
    {
        return this.depth;
    }

    /**
     * 添加一个匹配到的模式串（这个状态对应着这个模式串)
     *
     * @param keyword
     */
    public void addEmit(int keyword)
    {
        this.emits = keyword;
    }

    /**
     * 获取这个节点代表的模式串（们）
     *
     * @return
     */
    public Integer emit()
    {
        return this.emits;
    }

    /**
     * 是否是终止状态
     *
     * @return
     */
    public boolean isAcceptable()
    {
        return this.depth > 0 && this.emits != null;
    }

    /**
     * 转移到下一个状态
     *
     * @param character       希望按此字符转移
     * @param ignoreRootState 是否忽略根节点，如果是根节点自己调用则应该是true，否则为false
     * @return 转移结果
     */
    private State nextState(Integer character, boolean ignoreRootState)
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
     *
     * @param character
     * @return
     */
    public State nextState(Integer character)
    {
        return nextState(character, false);
    }

    /**
     * 按照character转移，任何节点转移失败会返回null
     *
     * @param character
     * @return
     */
    public State nextStateIgnoreRootState(Integer character)
    {
        return nextState(character, true);
    }

    public State addState(Integer character)
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

    public Set<Integer> getTransitions()
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
        sb.append('}');
        return sb.toString();
    }

    /**
     * 获取goto表
     *
     * @return
     */
    public Map<Integer, State> getSuccess()
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
