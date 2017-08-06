/**
 * 
 */
package com.hankcs.test.other;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

import com.hankcs.hanlp.HanLP;

/**
 * @author gonggawang
 *
 */
public class TestExtractSummary 
{
	private static final String str = "7月21日，渤海海况恶劣，至少发生3起沉船事故，10余名船员危在旦夕。危急时刻，中国海油渤海油田再次行动起来，紧急调配救援力量救起10名遇险人员。"
			+ "21日一早，一阵急促的铃声，在渤海石油管理局总值班室骤然响起。这是天津海上搜救中心打来的电话。正在值班的作业协调部主管邬礼凯心里“咯噔”一下——天津海上搜救中心称，"
			+ "在“海洋石油932”平台西南方7海里处，一艘货轮遇险、处于倾覆边缘，4名船员命悬一线。时间就是生命！邬礼凯立即组织海上救援力量，立即驰奔事故发生地点。"
			+ "“滨海264”船接到任务单后，仅一个小时便抵达事故现场。此时，货船已完全倾覆。“滨海264”立刻开展救援工作，仅25分钟便将4人全部救出。";
	
	private static final String separator = "[。?？!！]";
	
	@Test
	public void testExctractSummay() 
	{
		List<String> oldSum = HanLP.extractSummary(str, 2);
		List<String> newSum = HanLP.extractSummary(str, 2, separator);
		System.out.println("exctractSummay old:" + oldSum);
		System.out.println("exctractSummay new:" + newSum);
		
		assertTrue(oldSum.toString().length() < newSum.toString().length());
		assertFalse(oldSum.toString().contains("，"));
		assertTrue(newSum.toString().contains("，"));
	}
	
	@Test
	public void testGetSummary() 
	{
		
		String oldSum = HanLP.getSummary(str, 100);
		String newSum = HanLP.getSummary(str, 100, separator);
		
		System.out.println("getSummay old:" + oldSum);
		System.out.println("getSummay new:" + newSum);
		
		assertFalse(oldSum.contains("，"));
		assertTrue(newSum.contains("，"));
	}
	
}
