package com.hankcs.hanlp.algorithm.ahocorasick.interval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 线段树上面的节点，实际上是一些区间的集合，并且按中点维护了两个节点
 */
public class IntervalNode
{
    /**
     * 方向
     */
    private enum Direction
    {
        LEFT, RIGHT
    }

    /**
     * 区间集合的最左端
     */
    private IntervalNode left = null;
    /**
     * 最右端
     */
    private IntervalNode right = null;
    /**
     * 中点
     */
    private int point;
    /**
     * 区间集合
     */
    private List<Intervalable> intervals = new ArrayList<Intervalable>();

    /**
     * 构造一个节点
     * @param intervals
     */
    public IntervalNode(List<Intervalable> intervals)
    {
        this.point = determineMedian(intervals);

        List<Intervalable> toLeft = new ArrayList<Intervalable>();  // 以中点为界靠左的区间
        List<Intervalable> toRight = new ArrayList<Intervalable>(); // 靠右的区间

        for (Intervalable interval : intervals)
        {
            if (interval.getEnd() < this.point)
            {
                toLeft.add(interval);
            }
            else if (interval.getStart() > this.point)
            {
                toRight.add(interval);
            }
            else
            {
                this.intervals.add(interval);
            }
        }

        if (toLeft.size() > 0)
        {
            this.left = new IntervalNode(toLeft);
        }
        if (toRight.size() > 0)
        {
            this.right = new IntervalNode(toRight);
        }
    }

    /**
     * 计算中点
     * @param intervals 区间集合
     * @return 中点坐标
     */
    public int determineMedian(List<Intervalable> intervals)
    {
        int start = -1;
        int end = -1;
        for (Intervalable interval : intervals)
        {
            int currentStart = interval.getStart();
            int currentEnd = interval.getEnd();
            if (start == -1 || currentStart < start)
            {
                start = currentStart;
            }
            if (end == -1 || currentEnd > end)
            {
                end = currentEnd;
            }
        }
        return (start + end) / 2;
    }

    /**
     * 寻找与interval有重叠的区间
     * @param interval
     * @return
     */
    public List<Intervalable> findOverlaps(Intervalable interval)
    {

        List<Intervalable> overlaps = new ArrayList<Intervalable>();

        if (this.point < interval.getStart())
        {
            // 右边找找
            addToOverlaps(interval, overlaps, findOverlappingRanges(this.right, interval));
            addToOverlaps(interval, overlaps, checkForOverlapsToTheRight(interval));
        }
        else if (this.point > interval.getEnd())
        {
            // 左边找找
            addToOverlaps(interval, overlaps, findOverlappingRanges(this.left, interval));
            addToOverlaps(interval, overlaps, checkForOverlapsToTheLeft(interval));
        }
        else
        {
            // 否则在当前区间
            addToOverlaps(interval, overlaps, this.intervals);
            addToOverlaps(interval, overlaps, findOverlappingRanges(this.left, interval));
            addToOverlaps(interval, overlaps, findOverlappingRanges(this.right, interval));
        }

        return overlaps;
    }

    /**
     * 添加到重叠区间列表中
     * @param interval 跟此区间重叠
     * @param overlaps 重叠区间列表
     * @param newOverlaps 希望将这些区间加入
     */
    protected void addToOverlaps(Intervalable interval, List<Intervalable> overlaps, List<Intervalable> newOverlaps)
    {
        for (Intervalable currentInterval : newOverlaps)
        {
            if (!currentInterval.equals(interval))
            {
                overlaps.add(currentInterval);
            }
        }
    }

    /**
     * 往左边寻找重叠
     * @param interval
     * @return
     */
    protected List<Intervalable> checkForOverlapsToTheLeft(Intervalable interval)
    {
        return checkForOverlaps(interval, Direction.LEFT);
    }

    /**
     * 往右边寻找重叠
     * @param interval
     * @return
     */
    protected List<Intervalable> checkForOverlapsToTheRight(Intervalable interval)
    {
        return checkForOverlaps(interval, Direction.RIGHT);
    }

    /**
     * 寻找重叠
     * @param interval 一个区间，与该区间重叠
     * @param direction 方向，表明重叠区间在interval的左边还是右边
     * @return
     */
    protected List<Intervalable> checkForOverlaps(Intervalable interval, Direction direction)
    {

        List<Intervalable> overlaps = new ArrayList<Intervalable>();
        for (Intervalable currentInterval : this.intervals)
        {
            switch (direction)
            {
                case LEFT:
                    if (currentInterval.getStart() <= interval.getEnd())
                    {
                        overlaps.add(currentInterval);
                    }
                    break;
                case RIGHT:
                    if (currentInterval.getEnd() >= interval.getStart())
                    {
                        overlaps.add(currentInterval);
                    }
                    break;
            }
        }
        return overlaps;
    }

    /**
     * 是对IntervalNode.findOverlaps(Intervalable)的一个包装，防止NPE
     * @see com.hankcs.hanlp.algorithm.ahocorasick.interval.IntervalNode#findOverlaps(Intervalable)
     * @param node
     * @param interval
     * @return
     */
    protected static List<Intervalable> findOverlappingRanges(IntervalNode node, Intervalable interval)
    {
        if (node != null)
        {
            return node.findOverlaps(interval);
        }
        return Collections.emptyList();
    }

}
