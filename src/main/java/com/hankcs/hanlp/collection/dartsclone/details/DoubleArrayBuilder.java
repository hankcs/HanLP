/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hankcs.hanlp.collection.dartsclone.details;

/**
 * 双数组构建者
 *
 * @author
 */
public class DoubleArrayBuilder
{
    /**
     * 构建
     * @param keyset
     */
    public void build(Keyset keyset)
    {
        if (keyset.hasValues())
        {
            DawgBuilder dawgBuilder = new DawgBuilder();
            buildDawg(keyset, dawgBuilder);
            buildFromDawg(dawgBuilder);
            dawgBuilder.clear();
        }
        else
        {
            buildFromKeyset(keyset);
        }
    }

    public int[] copy()
    {
        int[] ret = new int[_units.size()];
        System.arraycopy(_units.getBuffer(), 0, ret, 0, _units.size());
        return ret;
    }

    void clear()
    {
        _units = null;
        _extras = null;
        _labels.clear();
        _table = null;
        _extrasHead = 0;
    }

    private static final int BLOCK_SIZE = 256;
    private static final int NUM_EXTRA_BLOCKS = 16;
    private static final int NUM_EXTRAS = BLOCK_SIZE * NUM_EXTRA_BLOCKS;

    private static final int UPPER_MASK = 0xFF << 21;
    private static final int LOWER_MASK = 0xFF;

    private static final int OFFSET_MASK = (1 << 31) | (1 << 8) | 0xFF;

    static class DoubleArrayBuilderExtraUnit
    {
        int prev;
        int next;
        boolean isFixed;
        boolean isUsed;
    }

    private int numBlocks()
    {
        return _units.size() / BLOCK_SIZE;
    }

    private DoubleArrayBuilderExtraUnit extras(int id)
    {
        return _extras[id % NUM_EXTRAS];
    }

    /**
     * 构建
     * @param keyset
     * @param dawgBuilder
     */
    private void buildDawg(Keyset keyset, DawgBuilder dawgBuilder)
    {
        dawgBuilder.init();
        for (int i = 0; i < keyset.numKeys(); ++i)
        {
            dawgBuilder.insert(keyset.getKey(i), keyset.getValue(i));
        }
        dawgBuilder.finish();
    }

    private void buildFromDawg(DawgBuilder dawg)
    {
        int numUnits = 1;
        while (numUnits < dawg.size())
        {
            numUnits <<= 1;
        }
        _units.reserve(numUnits);

        _table = new int[dawg.numIntersections()];
        _extras = new DoubleArrayBuilderExtraUnit[NUM_EXTRAS];
        for (int i = 0; i < _extras.length; ++i)
        {
            _extras[i] = new DoubleArrayBuilderExtraUnit();
        }

        reserveId(0);


        int[] units = _units.getBuffer();
        // _units[0].set_offset(1);
        units[0] |= 1 << 10;
        // _units[0].set_label(0);
        units[0] &= ~0xFF;

        if (dawg.child(dawg.root()) != 0)
        {
            buildFromDawg(dawg, dawg.root(), 0);
        }

        fixAllBlocks();

        _extras = null;
        _labels.clear();
        _table = null;
    }

    private void buildFromDawg(DawgBuilder dawg, int dawgId, int dictId)
    {
        int dawgChildId = dawg.child(dawgId);
        if (dawg.isIntersection(dawgChildId))
        {
            int intersectionId = dawg.intersectionId(dawgChildId);
            int offset = _table[intersectionId];
            int[] units = _units.getBuffer();
            if (offset != 0)
            {
                offset ^= dictId;
                if ((offset & UPPER_MASK) == 0 || (offset & LOWER_MASK) == 0)
                {
                    if (dawg.isLeaf(dawgChildId))
                    {
                        // units[dictId].setHasLeaf(true);
                        units[dictId] |= 1 << 8;
                    }
                    // units[dictId].setOffset(offset);
                    units[dictId] &= OFFSET_MASK;
                    units[dictId] |=
                            (offset < 1 << 21)
                                    ? offset << 10
                                    : (offset << 2) | (1 << 9);
                    return;
                }
            }
        }

        int offset = arrangeFromDawg(dawg, dawgId, dictId);
        if (dawg.isIntersection(dawgChildId))
        {
            _table[dawg.intersectionId(dawgChildId)] = offset;
        }

        do
        {
            byte childLabel = dawg.label(dawgChildId);
            int dictChildId = offset ^ (childLabel & 0xFF);
            if (childLabel != 0)
            {
                buildFromDawg(dawg, dawgChildId, dictChildId);
            }
            dawgChildId = dawg.sibling(dawgChildId);
        }
        while (dawgChildId != 0);
    }

    private int arrangeFromDawg(DawgBuilder dawg, int dawgId, int dictId)
    {
        _labels.resize(0);

        int dawgChildId = dawg.child(dawgId);
        while (dawgChildId != 0)
        {
            _labels.add(dawg.label(dawgChildId));
            dawgChildId = dawg.sibling(dawgChildId);
        }

        int offset = findValidOffset(dictId);
        int[] units = _units.getBuffer();
        // units[dictId].setOffset(dic_id ^ offset);
        units[dictId] &= OFFSET_MASK;
        int newId = dictId ^ offset;
        units[dictId] |=
                (newId < 1 << 21)
                        ? newId << 10
                        : (newId << 2) | (1 << 9);

        dawgChildId = dawg.child(dawgId);
        for (int i = 0; i < _labels.size(); ++i)
        {
            int dictChildId = offset ^ (_labels.get(i) & 0xFF);
            reserveId(dictChildId);
            units = _units.getBuffer();

            if (dawg.isLeaf(dawgChildId))
            {
                // units[dictId].setHasLeaf(true);
                units[dictId] |= 1 << 8;
                // units[dictChildId].setValue(dawg.value(dawgChildId));
                units[dictChildId] = dawg.value(dawgChildId) | (1 << 31);
            }
            else
            {
                // units[dictChildId].setLabel(_labels[i]);
                units[dictChildId] = (units[dictChildId] & ~0xFF)
                        | (_labels.get(i) & 0xFF);
            }

            dawgChildId = dawg.sibling(dawgChildId);
        }
        extras(offset).isUsed = true;

        return offset;
    }

    private void buildFromKeyset(Keyset keyset)
    {
        int numUnits = 1;
        while (numUnits < keyset.numKeys())
        {
            numUnits <<= 1;
        }
        _units.reserve(numUnits);

        _extras = new DoubleArrayBuilderExtraUnit[NUM_EXTRAS];
        for (int i = 0; i < _extras.length; ++i)
        {
            _extras[i] = new DoubleArrayBuilderExtraUnit();
        }

        reserveId(0);
        extras(0).isUsed = true;

        int[] units = _units.getBuffer();
        // units[0].setOffset(1);
        units[0] |= 1 << 10;
        // units[0].setLabel(0);
        units[0] &= ~0xFF;

        if (keyset.numKeys() > 0)
        {
            buildFromKeyset(keyset, 0, keyset.numKeys(), 0, 0);
        }

        fixAllBlocks();

        _extras = null;
        _labels.clear();
    }

    private void buildFromKeyset(Keyset keyset, int begin, int end, int depth,
                                 int dicId)
    {
        int offset = arrangeFromKeyset(keyset, begin, end, depth, dicId);

        while (begin < end)
        {
            if (keyset.getKeyByte(begin, depth) != 0)
            {
                break;
            }
            ++begin;
        }
        if (begin == end)
        {
            return;
        }

        int lastBegin = begin;
        byte lastLabel = keyset.getKeyByte(begin, depth);
        while (++begin < end)
        {
            byte label = keyset.getKeyByte(begin, depth);
            if (label != lastLabel)
            {
                buildFromKeyset(keyset, lastBegin, begin, depth + 1,
                                offset ^ (lastLabel & 0xFF));
                lastBegin = begin;
                lastLabel = keyset.getKeyByte(begin, depth);
            }
        }
        buildFromKeyset(keyset, lastBegin, end, depth + 1, offset ^ (lastLabel & 0xFF));
    }

    private int arrangeFromKeyset(Keyset keyset, int begin, int end, int depth,
                                  int dictId)
    {
        _labels.resize(0);

        int value = -1;
        for (int i = begin; i < end; ++i)
        {
            byte label = keyset.getKeyByte(i, depth);
            if (label == 0)
            {
                if (depth < keyset.getKey(i).length)
                {
                    throw new IllegalArgumentException(
                            "failed to build double-array: " +
                                    "invalid null character");
                }
                else if (keyset.getValue(i) < 0)
                {
                    throw new IllegalArgumentException(
                            "failed to build double-array: negative value");
                }

                if (value == -1)
                {
                    value = keyset.getValue(i);
                }
            }

            if (_labels.empty())
            {
                _labels.add(label);
            }
            else if (label != _labels.get(_labels.size() - 1))
            {
                if ((label & 0xFF) < (_labels.get(_labels.size() - 1) & 0xFF))
                {
                    throw new IllegalArgumentException(
                            "failed to build double-array: wrong key order");
                }
                _labels.add(label);
            }
        }

        int offset = findValidOffset(dictId);
        int[] units = _units.getBuffer();
        // units[dictId].setOffset(dictIad ^ offset);
        units[dictId] &= OFFSET_MASK;
        int newId = dictId ^ offset;
        units[dictId] |=
                (newId < 1 << 21)
                        ? newId << 10
                        : (newId << 2) | (1 << 9);

        for (int i = 0; i < _labels.size(); ++i)
        {
            int dictChildId = offset ^ (_labels.get(i) & 0xFF);
            reserveId(dictChildId);
            units = _units.getBuffer();
            if (_labels.get(i) == 0)
            {
                // units[dictId].setHasLeaf(true);
                units[dictId] |= 1 << 8;
                // units[dictChildId].setValue(value);
                units[dictChildId] = value | (1 << 31);
            }
            else
            {
                // units[dictChildId].setLabel(_labels[i]);
                units[dictChildId] = (units[dictChildId] & ~0xFF)
                        | (_labels.get(i) & 0xFF);
            }
        }
        extras(offset).isUsed = true;

        return offset;
    }

    int findValidOffset(int id)
    {
        if (_extrasHead >= _units.size())
        {
            return _units.size() | (id & LOWER_MASK);
        }

        int unfixedId = _extrasHead;
        do
        {
            int offset = unfixedId ^ (_labels.get(0) & 0xFF);
            if (isValidOffset(id, offset))
            {
                return offset;
            }
            unfixedId = extras(unfixedId).next;
        }
        while (unfixedId != _extrasHead);
        return _units.size() | (id & LOWER_MASK);
    }

    boolean isValidOffset(int id, int offset)
    {
        if (extras(offset).isUsed)
        {
            return false;
        }

        int relOffset = id ^ offset;
        if ((relOffset & LOWER_MASK) != 0 && (relOffset & UPPER_MASK) != 0)
        {
            return false;
        }

        for (int i = 1; i < _labels.size(); ++i)
        {
            if (extras(offset ^ (_labels.get(i) & 0xFF)).isFixed)
            {
                return false;
            }
        }

        return true;
    }

    void reserveId(int id)
    {
        if (id >= _units.size())
        {
            expandUnits();
        }

        if (id == _extrasHead)
        {
            _extrasHead = extras(id).next;
            if (_extrasHead == id)
            {
                _extrasHead = _units.size();
            }
        }
        extras(extras(id).prev).next = extras(id).next;
        extras(extras(id).next).prev = extras(id).prev;
        extras(id).isFixed = true;
    }

    void expandUnits()
    {
        int srcNumUnits = _units.size();
        int srcNumBlocks = numBlocks();

        int destNumUnits = srcNumUnits + BLOCK_SIZE;
        int destNumBlocks = srcNumBlocks + 1;

        if (destNumBlocks > NUM_EXTRA_BLOCKS)
        {
            fixBlock(srcNumBlocks - NUM_EXTRA_BLOCKS);
        }

        _units.resize(destNumUnits);

        if (destNumBlocks > NUM_EXTRA_BLOCKS)
        {
            for (int id = srcNumUnits; id < destNumUnits; ++id)
            {
                extras(id).isUsed = false;
                extras(id).isFixed = false;
            }
        }

        for (int i = srcNumUnits + 1; i < destNumUnits; ++i)
        {
            extras(i - 1).next = i;
            extras(i).prev = i - 1;
        }

        extras(srcNumUnits).prev = destNumUnits - 1;
        extras(destNumUnits - 1).next = srcNumUnits;

        extras(srcNumUnits).prev = extras(_extrasHead).prev;
        extras(destNumUnits - 1).next = _extrasHead;

        extras(extras(_extrasHead).prev).next = srcNumUnits;
        extras(_extrasHead).prev = destNumUnits - 1;
    }

    void fixAllBlocks()
    {
        int begin = 0;
        if (numBlocks() > NUM_EXTRA_BLOCKS)
        {
            begin = numBlocks() - NUM_EXTRA_BLOCKS;
        }
        int end = numBlocks();

        for (int blockId = begin; blockId != end; ++blockId)
        {
            fixBlock(blockId);
        }
    }

    void fixBlock(int blockId)
    {
        int begin = blockId * BLOCK_SIZE;
        int end = begin + BLOCK_SIZE;

        int unusedOffset = 0;
        for (int offset = begin; offset != end; ++offset)
        {
            if (!extras(offset).isUsed)
            {
                unusedOffset = offset;
                break;
            }
        }

        for (int id = begin; id != end; ++id)
        {
            if (!extras(id).isFixed)
            {
                reserveId(id);
                int[] units = _units.getBuffer();
                // units[id].setLabel(id ^ unused_offset);
                units[id] = (units[id] & ~0xFF)
                        | ((id ^ unusedOffset) & 0xFF);
            }
        }
    }

    private AutoIntPool _units = new AutoIntPool();
    private DoubleArrayBuilderExtraUnit[] _extras;
    private AutoBytePool _labels = new AutoBytePool();
    private int[] _table;
    private int _extrasHead;
}
