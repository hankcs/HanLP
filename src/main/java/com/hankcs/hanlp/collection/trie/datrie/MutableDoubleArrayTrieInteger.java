package com.hankcs.hanlp.collection.trie.datrie;

import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 可变双数组trie树，重构自：https://github.com/fancyerii/DoubleArrayTrie
 */
public class MutableDoubleArrayTrieInteger implements Serializable, Iterable<MutableDoubleArrayTrieInteger.KeyValuePair>, ICacheAble
{
    private static final long serialVersionUID = 5586394930559218802L;
    /**
     * 0x40000000
     */
    private static final int LEAF_BIT = 1073741824;
    private static final int[] EMPTY_WALK_STATE = {-1, -1};
    CharacterMapping charMap;
    /**
     * 字符串的终止字符（会在传入的字符串末尾添加该字符）
     */
    private static final char UNUSED_CHAR = '\000';
    /**
     * 终止字符的codePoint，这个字符作为叶节点的标识
     */
    private static final int UNUSED_CHAR_VALUE = UNUSED_CHAR;
    private IntArrayList check;
    private IntArrayList base;
    /**
     * 键值对数量
     */
    private int size;

    public MutableDoubleArrayTrieInteger(Map<String, Integer> stringIntegerMap)
    {
        this(stringIntegerMap.entrySet());
    }

    public MutableDoubleArrayTrieInteger(Set<Map.Entry<String, Integer>> entrySet)
    {
        this();
        for (Map.Entry<String, Integer> entry : entrySet)
        {
            put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * 激活指数膨胀
     *
     * @param exponentialExpanding
     */
    public void setExponentialExpanding(boolean exponentialExpanding)
    {
        check.setExponentialExpanding(exponentialExpanding);
        base.setExponentialExpanding(exponentialExpanding);
    }

    /**
     * 指数膨胀的底数
     *
     * @param exponentialExpandFactor
     */
    public void setExponentialExpandFactor(double exponentialExpandFactor)
    {
        check.setExponentialExpandFactor(exponentialExpandFactor);
        base.setExponentialExpandFactor(exponentialExpandFactor);
    }

    /**
     * 设置线性膨胀
     *
     * @param linearExpandFactor
     */
    public void setLinearExpandFactor(int linearExpandFactor)
    {
        check.setLinearExpandFactor(linearExpandFactor);
        base.setLinearExpandFactor(linearExpandFactor);
    }

    public MutableDoubleArrayTrieInteger()
    {
        this(new Utf8CharacterMapping());
    }

    public MutableDoubleArrayTrieInteger(CharacterMapping charMap)
    {
        this.charMap = charMap;
        clear();
    }

    public void clear()
    {
        this.base = new IntArrayList(this.charMap.getInitSize());
        this.check = new IntArrayList(this.charMap.getInitSize());

        this.base.append(0);
        this.check.append(0);

        this.base.append(1);
        this.check.append(0);
        expandArray(this.charMap.getInitSize());
    }

    public int getCheckArraySize()
    {
        return check.size();
    }

    public int getFreeSize()
    {
        int count = 0;
        int chk = this.check.get(0);
        while (chk != 0)
        {
            count++;
            chk = this.check.get(-chk);
        }

        return count;
    }

    private boolean isLeafValue(int value)
    {
        return (value > 0) && ((value & LEAF_BIT) != 0);
    }

    /**
     * 最高4位置1
     *
     * @param value
     * @return
     */
    private int setLeafValue(int value)
    {
        return value | LEAF_BIT;
    }

    /**
     * 最高4位置0
     *
     * @param value
     * @return
     */
    private int getLeafValue(int value)
    {
        return value ^ LEAF_BIT;
    }

    public int getBaseArraySize()
    {
        return this.base.size();
    }

    private int getBase(int index)
    {
        return this.base.get(index);
    }

    private int getCheck(int index)
    {
        return this.check.get(index);
    }

    private void setBase(int index, int value)
    {
        this.base.set(index, value);
    }

    private void setCheck(int index, int value)
    {
        this.check.set(index, value);
    }

    protected boolean isEmpty(int index)
    {
        return getCheck(index) <= 0;
    }

    private int getNextFreeBase(int nextChar)
    {
        int index = -getCheck(0);
        while (index != 0)
        {
            if (index > nextChar + 1) // 因为ROOT的index从1开始，所以至少要大于1
            {
                return index - nextChar;
            }
            index = -getCheck(index);
        }
        int oldSize = getBaseArraySize();
        expandArray(oldSize + this.base.getLinearExpandFactor());
        return oldSize;
    }

    private void addFreeLink(int index)
    {
        this.check.set(index, this.check.get(-this.base.get(0)));
        this.check.set(-this.base.get(0), -index);
        this.base.set(index, this.base.get(0));
        this.base.set(0, -index);
    }

    /**
     * 将index从空闲循环链表中删除
     *
     * @param index
     */
    private void deleteFreeLink(int index)
    {
        this.base.set(-this.check.get(index), this.base.get(index));
        this.check.set(-this.base.get(index), this.check.get(index));
    }

    /**
     * 动态数组扩容
     *
     * @param maxSize 需要的容量
     */
    private void expandArray(int maxSize)
    {
        int curSize = getBaseArraySize();
        if (curSize > maxSize)
        {
            return;
        }
        if (maxSize >= LEAF_BIT)
        {
            throw new RuntimeException("Double Array Trie size exceeds absolute threshold");
        }
        for (int i = curSize; i <= maxSize; ++i)
        {
            this.base.append(0);
            this.check.append(0);
            addFreeLink(i);
        }
    }

    /**
     * 插入条目
     *
     * @param key       键
     * @param value     值
     * @param overwrite 是否覆盖
     * @return
     */
    public boolean insert(String key, int value, boolean overwrite)
    {
        if ((null == key) || key.length() == 0 || (key.indexOf(UNUSED_CHAR) != -1))
        {
            return false;
        }
        if ((value < 0) || ((value & LEAF_BIT) != 0))
        {
            return false;
        }

        value = setLeafValue(value);

        int[] ids = this.charMap.toIdList(key + UNUSED_CHAR);

        int fromState = 1; // 根节点的index为1
        int toState = 1;
        int index = 0;
        while (index < ids.length)
        {
            int c = ids[index];
            toState = getBase(fromState) + c; // to = base[from] + c
            expandArray(toState);

            if (isEmpty(toState))
            {
                deleteFreeLink(toState);

                setCheck(toState, fromState); // check[to] = from
                if (index == ids.length - 1)  // Leaf
                {
                    ++this.size;
                    setBase(toState, value);  // base[to] = value
                }
                else
                {
                    int nextChar = ids[(index + 1)];
                    setBase(toState, getNextFreeBase(nextChar)); // base[to] = free_state - c
                }
            }
            else if (getCheck(toState) != fromState) // 冲突
            {
                solveConflict(fromState, c);
                continue;
            }
            fromState = toState;
            ++index;
        }
        if (overwrite)
        {
            setBase(toState, value);
        }
        return true;
    }

    /**
     * 寻找可以放下子节点集合的“连续”空闲区间
     *
     * @param children 子节点集合
     * @return base值
     */
    private int searchFreeBase(SortedSet<Integer> children)
    {
        int minChild = children.first();
        int maxChild = children.last();
        int current = 0;
        while (getCheck(current) != 0) // 循环链表回到了头，说明没有符合要求的“连续”区间
        {
            if (current > minChild + 1)
            {
                int base = current - minChild;
                boolean ok = true;
                for (Iterator<Integer> it = children.iterator(); it.hasNext(); ) // 检查是否每个子节点的位置都空闲（“连续”区间）
                {
                    int to = base + it.next();
                    if (to >= getBaseArraySize())
                    {
                        ok = false;
                        break;
                    }
                    if (!isEmpty(to))
                    {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                {
                    return base;
                }
            }
            current = -getCheck(current); // 从链表中取出下一个空闲位置
        }
        int oldSize = getBaseArraySize(); // 没有足够长的“连续”空闲区间，所以在双数组尾部额外分配一块
        expandArray(oldSize + maxChild);
        return oldSize;
    }

    /**
     * 解决冲突
     *
     * @param parent   父节点
     * @param newChild 子节点的char值
     */
    private void solveConflict(int parent, int newChild)
    {
        // 找出parent的所有子节点
        TreeSet<Integer> children = new TreeSet<Integer>();
        children.add(newChild);
        final int charsetSize = this.charMap.getCharsetSize();
        for (int c = 0; c < charsetSize; ++c)
        {
            int next = getBase(parent) + c;
            if (next >= getBaseArraySize())
            {
                break;
            }
            if (getCheck(next) == parent)
            {
                children.add(c);
            }
        }
        // 移动旧子节点到新的位置
        int newBase = searchFreeBase(children);
        children.remove(newChild);
        for (Integer c : children)
        {
            int child = newBase + c;
            deleteFreeLink(child);

            setCheck(child, parent);
            int childBase = getBase(getBase(parent) + c);
            setBase(child, childBase);

            if (!isLeafValue(childBase))
            {
                for (int d = 0; d < charsetSize; ++d)
                {
                    int to = childBase + d;
                    if (to >= getBaseArraySize())
                    {
                        break;
                    }
                    if (getCheck(to) == getBase(parent) + c)
                    {
                        setCheck(to, child);
                    }
                }
            }
            addFreeLink(getBase(parent) + c);
        }
        // 更新新base值
        setBase(parent, newBase);
    }

    /**
     * 键值对个数
     *
     * @return
     */
    public int size()
    {
        return this.size;
    }

    public boolean isEmpty()
    {
        return size == 0;
    }

    /**
     * 覆盖模式添加
     *
     * @param key
     * @param value
     * @return
     */
    public boolean insert(String key, int value)
    {
        return insert(key, value, true);
    }

    /**
     * 非覆盖模式添加
     *
     * @param key
     * @param value
     * @return
     */
    public boolean add(String key, int value)
    {
        return insert(key, value, false);
    }

    /**
     * 非覆盖模式添加，值默认为当前集合大小
     *
     * @param key
     * @return
     */
    public boolean add(String key)
    {
        return add(key, size);
    }

    /**
     * 查询以prefix开头的所有键
     *
     * @param prefix
     * @return
     */
    public List<String> prefixMatch(String prefix)
    {
        int curState = 1;
        IntArrayList bytes = new IntArrayList(prefix.length() * 4);
        for (int i = 0; i < prefix.length(); i++)
        {
            int codePoint = prefix.charAt(i);
            if (curState < 1)
            {
                return Collections.emptyList();
            }
            if ((curState != 1) && (isEmpty(curState)))
            {
                return Collections.emptyList();
            }
            int[] ids = this.charMap.toIdList(codePoint);
            if (ids.length == 0)
            {
                return Collections.emptyList();
            }
            for (int j = 0; j < ids.length; j++)
            {
                int c = ids[j];
                if ((getBase(curState) + c < getBaseArraySize())
                    && (getCheck(getBase(curState) + c) == curState))
                {
                    bytes.append(c);
                    curState = getBase(curState) + c;
                }
                else
                {
                    return Collections.emptyList();
                }
            }

        }
        List<String> result = new ArrayList<String>();
        recursiveAddSubTree(curState, result, bytes);

        return result;
    }

    private void recursiveAddSubTree(int curState, List<String> result, IntArrayList bytes)
    {
        if (getCheck(getBase(curState) + UNUSED_CHAR_VALUE) == curState)
        {
            byte[] array = new byte[bytes.size()];
            for (int i = 0; i < bytes.size(); i++)
            {
                array[i] = (byte) bytes.get(i);
            }
            result.add(new String(array, Utf8CharacterMapping.UTF_8));
        }
        int base = getBase(curState);
        for (int c = 0; c < charMap.getCharsetSize(); c++)
        {
            if (c == UNUSED_CHAR_VALUE) continue;
            int check = getCheck(base + c);
            if (base + c < getBaseArraySize() && check == curState)
            {
                bytes.append(c);
                recursiveAddSubTree(base + c, result, bytes);
                bytes.removeLast();
            }
        }
    }

    /**
     * 最长查询
     *
     * @param query
     * @param start
     * @return (最长长度，对应的值)
     */
    public int[] findLongest(CharSequence query, int start)
    {
        if ((query == null) || (start >= query.length()))
        {
            return new int[]{0, -1};
        }
        int state = 1;
        int maxLength = 0;
        int lastVal = -1;
        for (int i = start; i < query.length(); i++)
        {
            int[] res = transferValues(state, query.charAt(i));
            if (res[0] == -1)
            {
                break;
            }
            state = res[0];
            if (res[1] != -1)
            {
                maxLength = i - start + 1;
                lastVal = res[1];
            }
        }
        return new int[]{maxLength, lastVal};
    }

    public int[] findWithSupplementary(String query, int start)
    {
        if ((query == null) || (start >= query.length()))
        {
            return new int[]{0, -1};
        }
        int curState = 1;
        int maxLength = 0;
        int lastVal = -1;
        int charCount = 1;
        for (int i = start; i < query.length(); i += charCount)
        {
            int codePoint = query.codePointAt(i);
            charCount = Character.charCount(codePoint);
            int[] res = transferValues(curState, codePoint);
            if (res[0] == -1)
            {
                break;
            }
            curState = res[0];
            if (res[1] != -1)
            {
                maxLength = i - start + 1;
                lastVal = res[1];
            }
        }
        return new int[]{maxLength, lastVal};

    }

    public List<int[]> findAllWithSupplementary(String query, int start)
    {
        List<int[]> ret = new ArrayList<int[]>(5);
        if ((query == null) || (start >= query.length()))
        {
            return ret;
        }
        int curState = 1;
        int charCount = 1;
        for (int i = start; i < query.length(); i += charCount)
        {
            int codePoint = query.codePointAt(i);
            charCount = Character.charCount(codePoint);
            int[] res = transferValues(curState, codePoint);
            if (res[0] == -1)
            {
                break;
            }
            curState = res[0];
            if (res[1] != -1)
            {
                ret.add(new int[]{i - start + 1, res[1]});
            }
        }
        return ret;
    }

    /**
     * 查询与query的前缀重合的所有词语
     *
     * @param query
     * @param start
     * @return
     */
    public List<int[]> commonPrefixSearch(String query, int start)
    {
        List<int[]> ret = new ArrayList<int[]>(5);
        if ((query == null) || (start >= query.length()))
        {
            return ret;
        }
        int curState = 1;
        for (int i = start; i < query.length(); i++)
        {
            int[] res = transferValues(curState, query.charAt(i));
            if (res[0] == -1)
            {
                break;
            }
            curState = res[0];
            if (res[1] != -1)
            {
                ret.add(new int[]{i - start + 1, res[1]});
            }
        }
        return ret;
    }

    /**
     * 转移状态并输出值
     *
     * @param state
     * @param codePoint char
     * @return
     */
    public int[] transferValues(int state, int codePoint)
    {
        if (state < 1)
        {
            return EMPTY_WALK_STATE;
        }
        if ((state != 1) && (isEmpty(state)))
        {
            return EMPTY_WALK_STATE;
        }
        int[] ids = this.charMap.toIdList(codePoint);
        if (ids.length == 0)
        {
            return EMPTY_WALK_STATE;
        }
        for (int i = 0; i < ids.length; i++)
        {
            int c = ids[i];
            if ((getBase(state) + c < getBaseArraySize())
                && (getCheck(getBase(state) + c) == state))
            {
                state = getBase(state) + c;
            }
            else
            {
                return EMPTY_WALK_STATE;
            }
        }
        if (getCheck(getBase(state) + UNUSED_CHAR_VALUE) == state)
        {
            int value = getLeafValue(getBase(getBase(state)
                                                 + UNUSED_CHAR_VALUE));
            return new int[]{state, value};
        }
        return new int[]{state, -1};
    }

    /**
     * 转移状态
     *
     * @param state
     * @param codePoint
     * @return
     */
    public int transfer(int state, int codePoint)
    {
        if (state < 1)
        {
            return -1;
        }
        if ((state != 1) && (isEmpty(state)))
        {
            return -1;
        }
        int[] ids = this.charMap.toIdList(codePoint);
        if (ids.length == 0)
        {
            return -1;
        }
        return transfer(state, ids);
    }

    /**
     * 转移状态
     *
     * @param state
     * @param ids
     * @return
     */
    private int transfer(int state, int[] ids)
    {
        for (int c : ids)
        {
            if ((getBase(state) + c < getBaseArraySize())
                && (getCheck(getBase(state) + c) == state))
            {
                state = getBase(state) + c;
            }
            else
            {
                return -1;
            }
        }
        return state;
    }

    public int stateValue(int state)
    {
        int leaf = getBase(state) + UNUSED_CHAR_VALUE;
        if (getCheck(leaf) == state)
        {
            return getLeafValue(getBase(leaf));
        }
        return -1;
    }

    /**
     * 去掉多余的buffer
     */
    public void loseWeight()
    {
        base.loseWeight();
        check.loseWeight();
    }

    /**
     * 将值大于等于from的统一递减1<br>
     *
     * @param from
     */
    public void decreaseValues(int from)
    {
        for (int state = 1; state < getBaseArraySize(); ++state)
        {
            int leaf = getBase(state) + UNUSED_CHAR_VALUE;
            if (1 < leaf && leaf < getCheckArraySize() && getCheck(leaf) == state)
            {
                int value = getLeafValue(getBase(leaf));
                if (value >= from)
                {
                    setBase(leaf, setLeafValue(--value));
                }
            }
        }
    }

    /**
     * 精确查询
     *
     * @param key
     * @param start
     * @return -1表示不存在
     */
    public int get(String key, int start)
    {
        assert key != null;
        assert 0 <= start && start <= key.length();
        int state = 1;
        int[] ids = charMap.toIdList(key.substring(start));
        state = transfer(state, ids);
        if (state < 0)
        {
            return -1;
        }
        return stateValue(state);
    }

    /**
     * 精确查询
     *
     * @param key
     * @return -1表示不存在
     */
    public int get(String key)
    {
        return get(key, 0);
    }

    /**
     * 设置键值 （同put）
     *
     * @param key
     * @param value
     * @return 是否设置成功（失败的原因是键值不合法）
     */
    public boolean set(String key, int value)
    {
        return insert(key, value, true);
    }

    /**
     * 设置键值 （同set）
     *
     * @param key
     * @param value
     * @return 是否设置成功（失败的原因是键值不合法）
     */
    public boolean put(String key, int value)
    {
        return insert(key, value, true);
    }

    /**
     * 删除键
     *
     * @param key
     * @return 值
     */
    public int remove(String key)
    {
        return delete(key);
    }

    /**
     * 删除键
     *
     * @param key
     * @return 值
     */
    public int delete(String key)
    {
        if (key == null)
        {
            return -1;
        }
        int curState = 1;
        int[] ids = this.charMap.toIdList(key);

        int[] path = new int[ids.length + 1];
        int i = 0;
        for (; i < ids.length; i++)
        {
            int c = ids[i];
            if ((getBase(curState) + c >= getBaseArraySize())
                || (getCheck(getBase(curState) + c) != curState))
            {
                break;
            }
            curState = getBase(curState) + c;
            path[i] = curState;
        }
        int ret = -1;
        if (i == ids.length)
        {
            if (getCheck(getBase(curState) + UNUSED_CHAR_VALUE) == curState)
            {
                --this.size;
                ret = getLeafValue(getBase(getBase(curState) + UNUSED_CHAR_VALUE));
                path[(path.length - 1)] = (getBase(curState) + UNUSED_CHAR_VALUE);
                for (int j = path.length - 1; j >= 0; --j)
                {
                    boolean isLeaf = true;
                    int state = path[j];
                    for (int k = 0; k < this.charMap.getCharsetSize(); k++)
                    {
                        if (isLeafValue(getBase(state)))
                        {
                            break;
                        }
                        if ((getBase(state) + k < getBaseArraySize())
                            && (getCheck(getBase(state) + k) == state))
                        {
                            isLeaf = false;
                            break;
                        }
                    }
                    if (!isLeaf)
                    {
                        break;
                    }
                    addFreeLink(state);
                }
            }
        }
        return ret;
    }

    /**
     * 获取空闲的数组元素个数
     *
     * @return
     */
    public int getEmptySize()
    {
        int size = 0;
        for (int i = 0; i < getBaseArraySize(); i++)
        {
            if (isEmpty(i))
            {
                ++size;
            }
        }
        return size;
    }

    /**
     * 可以设置的最大值
     *
     * @return
     */
    public int getMaximumValue()
    {
        return LEAF_BIT - 1;
    }

    public Set<Map.Entry<String, Integer>> entrySet()
    {
        return new Set<Map.Entry<String, Integer>>()
        {
            @Override
            public int size()
            {
                return MutableDoubleArrayTrieInteger.this.size;
            }

            @Override
            public boolean isEmpty()
            {
                return MutableDoubleArrayTrieInteger.this.isEmpty();
            }

            @Override
            public boolean contains(Object o)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public Iterator<Map.Entry<String, Integer>> iterator()
            {
                return new Iterator<Map.Entry<String, Integer>>()
                {
                    KeyValuePair iterator = MutableDoubleArrayTrieInteger.this.iterator();

                    @Override
                    public boolean hasNext()
                    {
                        return iterator.hasNext();
                    }

                    @Override
                    public void remove()
                    {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public Map.Entry<String, Integer> next()
                    {
                        iterator.next();
                        return new AbstractMap.SimpleEntry<String, Integer>(iterator.key, iterator.value);
                    }
                };
            }

            @Override
            public Object[] toArray()
            {
                ArrayList<Map.Entry<String, Integer>> entries = new ArrayList<Map.Entry<String, Integer>>(size);
                for (Map.Entry<String, Integer> entry : this)
                {
                    entries.add(entry);
                }
                return entries.toArray();
            }

            @Override
            public <T> T[] toArray(T[] a)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean add(Map.Entry<String, Integer> stringIntegerEntry)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean remove(Object o)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean containsAll(Collection<?> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean addAll(Collection<? extends Map.Entry<String, Integer>> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean retainAll(Collection<?> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean removeAll(Collection<?> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public void clear()
            {
                throw new UnsupportedOperationException();
            }
        };
    }

    @Override
    public KeyValuePair iterator()
    {
        return new KeyValuePair();
    }

    public boolean containsKey(String key)
    {
        return get(key) != -1;
    }

    public Set<String> keySet()
    {
        return new Set<String>()
        {
            @Override
            public int size()
            {
                return MutableDoubleArrayTrieInteger.this.size;
            }

            @Override
            public boolean isEmpty()
            {
                return MutableDoubleArrayTrieInteger.this.isEmpty();
            }

            @Override
            public boolean contains(Object o)
            {
                return MutableDoubleArrayTrieInteger.this.containsKey((String) o);
            }

            @Override
            public Iterator<String> iterator()
            {
                return new Iterator<String>()
                {
                    KeyValuePair iterator = MutableDoubleArrayTrieInteger.this.iterator();

                    @Override
                    public void remove()
                    {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public boolean hasNext()
                    {
                        return iterator.hasNext();
                    }

                    @Override
                    public String next()
                    {
                        return iterator.next().key;
                    }
                };
            }

            @Override
            public Object[] toArray()
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public <T> T[] toArray(T[] a)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean add(String s)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean remove(Object o)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean containsAll(Collection<?> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean addAll(Collection<? extends String> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean retainAll(Collection<?> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean removeAll(Collection<?> c)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public void clear()
            {
                throw new UnsupportedOperationException();
            }
        };
    }

    @Override
    public void save(DataOutputStream out) throws IOException
    {
        if (!(charMap instanceof Utf8CharacterMapping))
        {
            logger.warning("将来需要在构造的时候传入 " + charMap.getClass());
        }
        out.writeInt(size);
        base.save(out);
        check.save(out);
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        size = byteArray.nextInt();
        if (!base.load(byteArray)) return false;
        if (!check.load(byteArray)) return false;
        return true;
    }

    private void writeObject(ObjectOutputStream out) throws IOException
    {
        out.writeInt(size);
        out.writeObject(base);
        out.writeObject(check);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
    {
        size = in.readInt();
        base = (IntArrayList) in.readObject();
        check = (IntArrayList) in.readObject();
        charMap = new Utf8CharacterMapping();
    }

//    /**
//     * 遍历时无法删除
//     *
//     * @return
//     */
//    public DATIterator iterator()
//    {
//        return new KeyValuePair();
//    }

    public class KeyValuePair implements Iterator<KeyValuePair>
    {
        /**
         * 储存(index, charPoint)
         */
        private IntArrayList path;
        /**
         * 当前所处的键值的索引
         */
        private int index;
        private int value = -1;
        private String key = null;
        private int currentBase;

        public KeyValuePair()
        {
            path = new IntArrayList(20);
            path.append(1); // ROOT
            int from = 1;
            int b = base.get(from);
            if (size > 0)
            {
                while (true)
                {
                    for (int i = 0; i < charMap.getCharsetSize(); i++)
                    {
                        int c = check.get(b + i);
                        if (c == from)
                        {
                            path.append(i);
                            from = b + i;
                            path.append(from);
                            b = base.get(from);
                            i = 0;
                            if (getCheck(b + UNUSED_CHAR_VALUE) == from)
                            {
                                value = getLeafValue(getBase(b + UNUSED_CHAR_VALUE));
                                int[] ids = new int[path.size() / 2];
                                for (int k = 0, j = 1; j < path.size(); k++, j += 2)
                                {
                                    ids[k] = path.get(j);
                                }
                                key = charMap.toString(ids);
                                path.append(UNUSED_CHAR_VALUE);
                                currentBase = b;
                                return;
                            }
                        }
                    }
                }
            }
        }

        public String key()
        {
            return key;
        }

        public int value()
        {
            return value;
        }

        public String getKey()
        {
            return key;
        }

        public int getValue()
        {
            return value;
        }

        public int setValue(int v)
        {
            int value = getLeafValue(v);
            setBase(currentBase + UNUSED_CHAR_VALUE, value);
            this.value = v;
            return v;
        }

        @Override
        public boolean hasNext()
        {
            return index < size;
        }

        @Override
        public KeyValuePair next()
        {
            if (index >= size)
            {
                throw new NoSuchElementException();
            }
            else if (index == 0)
            {
            }
            else
            {
                while (path.size() > 0)
                {
                    int charPoint = path.pop();
                    int base = path.getLast();
                    int n = getNext(base, charPoint);
                    if (n != -1) break;
                    path.removeLast();
                }
            }

            ++index;
            return this;
        }

        @Override
        public void remove()
        {
            throw new UnsupportedOperationException();
        }

        /**
         * 遍历下一个终止路径
         *
         * @param parent    父节点
         * @param charPoint 子节点的char
         * @return
         */
        private int getNext(int parent, int charPoint)
        {
            int startChar = charPoint + 1;
            int baseParent = getBase(parent);
            int from = parent;

            for (int i = startChar; i < charMap.getCharsetSize(); i++)
            {
                int to = baseParent + i;
                if (check.size() > to && check.get(to) == from)
                {
                    path.append(i);
                    from = to;
                    path.append(from);
                    baseParent = base.get(from);
                    if (getCheck(baseParent + UNUSED_CHAR_VALUE) == from)
                    {
                        value = getLeafValue(getBase(baseParent + UNUSED_CHAR_VALUE));
                        int[] ids = new int[path.size() / 2];
                        for (int k = 0, j = 1; j < path.size(); ++k, j += 2)
                        {
                            ids[k] = path.get(j);
                        }
                        key = charMap.toString(ids);
                        path.append(UNUSED_CHAR_VALUE);
                        currentBase = baseParent;
                        return from;
                    }
                    else
                    {
                        return getNext(from, 0);
                    }
                }
            }
            return -1;
        }

        @Override
        public String toString()
        {
            return key + '=' + value;
        }
    }

}
