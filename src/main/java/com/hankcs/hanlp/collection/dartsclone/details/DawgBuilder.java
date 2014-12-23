/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone.details;

import java.util.ArrayList;

/**
 * 有向无环字图
 * @author
 */
class DawgBuilder
{
    /**
     * 根节点id
     * @return 0
     */
    int root()
    {
        return 0;
    }

    /**
     * 获取节点的孩子
     * @param id 节点的id
     * @return 孩子的id
     */
    int child(int id)
    {
        // return _units.get(id).child();
        return _units.get(id) >>> 2;
    }

    /**
     * 获取兄弟节点
     * @param id 兄弟节点的id
     * @return 下一个兄弟节点的id，或者0表示没有兄弟节点
     */
    int sibling(int id)
    {
        // return _units.get(id).hasSibling() ? (id + 1) : 0;
        return ((_units.get(id) & 1) == 1) ? (id + 1) : 0;
    }

    /**
     * 获取值
     * @param id 节点id
     * @return 节点的值
     */
    int value(int id)
    {
        // return _units.get(id).value();
        return _units.get(id) >>> 1;
    }

    /**
     * 是否是叶子节点
     * @param id 节点id
     * @return 是否是叶子节点
     */
    boolean isLeaf(int id)
    {
        return label(id) == 0;
    }

    /**
     * 获取label
     * @param id 节点的id
     * @return
     */
    byte label(int id)
    {
        return _labels.get(id);
    }

    /**
     * 是否是分叉点
     * @param id 节点id
     * @return
     */
    boolean isIntersection(int id)
    {
        return _isIntersections.get(id);
    }

    int intersectionId(int id)
    {
        return _isIntersections.rank(id) - 1;
    }

    int numIntersections()
    {
        return _isIntersections.numOnes();
    }

    int size()
    {
        return _units.size();
    }

    /**
     * 初始化
     */
    void init()
    {
        _table.resize(INITIAL_TABLE_SIZE, 0);

        appendNode();
        appendUnit();

        _numStates = 1;

        _nodes.get(0).label = (byte) 0xFF;
        _nodeStack.add(0);
    }

    void finish()
    {
        flush(0);

        _units.set(0, _nodes.get(0).unit());
        _labels.set(0, _nodes.get(0).label);

        _nodes.clear();
        _table.clear();
        _nodeStack.clear();
        _recycleBin.clear();

        _isIntersections.build();
    }

    void insert(byte[] key, int value)
    {
        if (value < 0)
        {
            throw new IllegalArgumentException(
                    "failed to insert key: negative value");
        }
        if (key.length == 0)
        {
            throw new IllegalArgumentException(
                    "failed to inset key: zero-length key");
        }

        int id = 0;
        int keyPos = 0;

        for (; keyPos <= key.length; ++keyPos)
        {
            int childId = _nodes.get(id).child;
            if (childId == 0)
            {
                break;
            }

            byte keyLabel = keyPos < key.length ? key[keyPos] : 0;
            if (keyPos < key.length && keyLabel == 0)
            {
                throw new IllegalArgumentException(
                        "failed to insert key: invalid null character");
            }

            byte unitLabel = _nodes.get(childId).label;
            if ((keyLabel & 0xFF) < (unitLabel & 0xFF))
            {
                throw new IllegalArgumentException(
                        "failed to insert key: wrong key order");
            }
            else if ((keyLabel & 0xFF) > (unitLabel & 0xFF))
            {
                _nodes.get(childId).hasSibling = true;
                flush(childId);
                break;
            }
            id = childId;
        }

        if (keyPos > key.length)
        {
            return;
        }

        for (; keyPos <= key.length; ++keyPos)
        {
            byte keyLabel = (keyPos < key.length) ? key[keyPos] : 0;
            int childId = appendNode();

            DawgNode node = _nodes.get(id);
            DawgNode child = _nodes.get(childId);

            if (node.child == 0)
            {
                child.isState = true;
            }
            child.sibling = node.child;
            child.label = keyLabel;
            node.child = childId;
            _nodeStack.add(childId);

            id = childId;
        }
        _nodes.get(id).setValue(value);
    }

    void clear()
    {
        _nodes.clear();
        _units.clear();
        _labels.clear();
        _isIntersections.clear();
        _table.clear();
        _nodeStack.clear();
        _recycleBin.clear();
        _numStates = 0;
    }

    static class DawgNode
    {
        int child;
        int sibling;
        byte label;
        boolean isState;
        boolean hasSibling;

        void reset()
        {
            child = 0;
            sibling = 0;
            label = (byte) 0;
            isState = false;
            hasSibling = false;
        }

        int getValue()
        {
            return child;
        }

        void setValue(int value)
        {
            child = value;
        }

        int unit()
        {
            if (label == 0)
            {
                return (child << 1) | (hasSibling ? 1 : 0);
            }
            return (child << 2) | (isState ? 2 : 0) | (hasSibling ? 1 : 0);
        }
    }

    private void flush(int id)
    {
        while (_nodeStack.get(_nodeStack.size() - 1) != id)
        {
            int nodeId = _nodeStack.get(_nodeStack.size() - 1);
            _nodeStack.deleteLast();

            if (_numStates >= _table.size() - (_table.size() >>> 2))
            {
                expandTable();
            }

            int numSiblings = 0;
            for (int i = nodeId; i != 0; i = _nodes.get(i).sibling)
            {
                ++numSiblings;
            }

            // make an array of length 1 to emulate pass-by-reference
            int[] matchHashId = findNode(nodeId);
            int matchId = matchHashId[0];
            int hashId = matchHashId[1];

            if (matchId != 0)
            {
                _isIntersections.set(matchId, true);
            }
            else
            {
                int unitId = 0;
                for (int i = 0; i < numSiblings; ++i)
                {
                    unitId = appendUnit();
                }
                for (int i = nodeId; i != 0; i = _nodes.get(i).sibling)
                {
                    _units.set(unitId, _nodes.get(i).unit());
                    _labels.set(unitId, _nodes.get(i).label);
                    --unitId;
                }
                matchId = unitId + 1;
                _table.set(hashId, matchId);
                ++_numStates;
            }

            for (int i = nodeId, next; i != 0; i = next)
            {
                next = _nodes.get(i).sibling;
                freeNode(i);
            }

            _nodes.get(_nodeStack.get(_nodeStack.size() - 1)).child = matchId;
        }
        _nodeStack.deleteLast();
    }

    private void expandTable()
    {
        int tableSize = _table.size() << 1;
        _table.clear();
        _table.resize(tableSize, 0);

        for (int id = 1; id < _units.size(); ++id)
        {
//            if (_labels.get(i) == 0 || _units.get(id).isState)) {
            if (_labels.get(id) == 0 || (_units.get(id) & 2) == 2)
            {
                int[] ret = findUnit(id);
                int hashId = ret[1];
                _table.set(hashId, id);
            }
        }
    }

    private int[] findUnit(int id)
    {
        int[] ret = new int[2];
        int hashId = hashUnit(id) % _table.size();
        for (; ; hashId = (hashId + 1) % _table.size())
        {
            // Remainder adjustment.
            if (hashId < 0)
            {
                hashId += _table.size();
            }
            int unitId = _table.get(hashId);
            if (unitId == 0)
            {
                break;
            }

            // there must not be the same unit.
        }
        ret[1] = hashId;
        return ret;
    }

    private int[] findNode(int nodeId)
    {
        int[] ret = new int[2];
        int hashId = hashNode(nodeId) % _table.size();
        for (; ; hashId = (hashId + 1) % _table.size())
        {
            // Remainder adjustment
            if (hashId < 0)
            {
                hashId += _table.size();
            }
            int unitId = _table.get(hashId);
            if (unitId == 0)
            {
                break;
            }

            if (areEqual(nodeId, unitId))
            {
                ret[0] = unitId;
                ret[1] = hashId;
                return ret;
            }
        }
        ret[1] = hashId;
        return ret;
    }

    private boolean areEqual(int nodeId, int unitId)
    {
        for (int i = _nodes.get(nodeId).sibling; i != 0;
             i = _nodes.get(i).sibling)
        {
//            if (_units.get(unitId).hasSibling() == false) {
            if ((_units.get(unitId) & 1) != 1)
            {
                return false;
            }
            ++unitId;
        }
//        if (_units.get(unitId).hasSibling() == true) {
        if ((_units.get(unitId) & 1) == 1)
        {
            return false;
        }

        for (int i = nodeId; i != 0; i = _nodes.get(i).sibling, --unitId)
        {
//            if (_nodes.get(i) != _units.get(unitId).unit() ||
            if (_nodes.get(i).unit() != _units.get(unitId) ||
                    _nodes.get(i).label != _labels.get(unitId))
            {
                return false;
            }
        }
        return true;
    }

    private int hashUnit(int id)
    {
        int hashValue = 0;
        for (; id != 0; ++id)
        {
//            int unit = _units.get(id).unit();
            int unit = _units.get(id);
            byte label = _labels.get(id);
            hashValue ^= hash(((label & 0xFF) << 24) ^ unit);

//            if (_units.get(id).hasSibling() == false) {
            if ((_units.get(id) & 1) != 1)
            {
                break;
            }
        }
        return hashValue;
    }

    private int hashNode(int id)
    {
        int hashValue = 0;
        for (; id != 0; id = _nodes.get(id).sibling)
        {
            int unit = _nodes.get(id).unit();
            byte label = _nodes.get(id).label;
            hashValue ^= hash(((label & 0xFF) << 24) ^ unit);
        }
        return hashValue;
    }

    private int appendUnit()
    {
        _isIntersections.append();
        _units.add(0);
        _labels.add((byte) 0);

        return _isIntersections.size() - 1;
    }

    private int appendNode()
    {
        int id;
        if (_recycleBin.empty())
        {
            id = _nodes.size();
            _nodes.add(new DawgNode());
        }
        else
        {
            id = _recycleBin.get(_recycleBin.size() - 1);
            _nodes.get(id).reset();
            _recycleBin.deleteLast();
        }
        return id;
    }

    private void freeNode(int id)
    {
        _recycleBin.add(id);
    }

    private static int hash(int key)
    {
        key = ~key + (key << 15);  // key = (key << 15) - key - 1;
        key = key ^ (key >>> 12);
        key = key + (key << 2);
        key = key ^ (key >>> 4);
        key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
        key = key ^ (key >>> 16);
        return key;
    }

    private static final int INITIAL_TABLE_SIZE = 1 << 10;
    private ArrayList<DawgNode> _nodes = new ArrayList<DawgNode>();
    private AutoIntPool _units = new AutoIntPool();
    private AutoBytePool _labels = new AutoBytePool();
    private BitVector _isIntersections = new BitVector();
    private AutoIntPool _table = new AutoIntPool();
    private AutoIntPool _nodeStack = new AutoIntPool();
    private AutoIntPool _recycleBin = new AutoIntPool();
    private int _numStates;
}
