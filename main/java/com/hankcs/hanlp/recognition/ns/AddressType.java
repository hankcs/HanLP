package com.hankcs.hanlp.recognition.ns;

/**
 * 地址的各个类型
 */
public enum AddressType
{
    /**
     * 国家
     */
    Country,
    /**
     * 直辖市
     */
    Municipality,
    /**
     * 特别行政区后缀
     */
    SuffixMunicipality,
    /**
     * 省
     */
    Province,
    /**
     * 市
     */
    City,
    /**
     * 区
     */
    County,
    /**
     * 镇
     */
    Town,
    /**
     * 街
     */
    Street,
    /**
     * 街门牌号
     */
    StreetNo,
    /**
     * 编号
     */
    No,
    /**
     * 字母符号
     */
    Symbol,
    /**
     * 地标建筑 例如 ** 大厦  门牌设施
     */
    LandMark,
    /**
     * 相对位置
     */
    RelatedPos,
    /**
     * 交叉路
     */
    Crossing,
    /**
     * 详细描述
     */
    DetailDesc,
    /**
     * 子设施
     */
    childFacility,
    /**
     * 村
     */
    Village,
    /**
     * 村编号
     */
    VillageNo,
    /**
     * 楼号
     */
    BuildingNo,
    /**
     * 楼单元
     */
    BuildingUnit,
    /**
     * 楼单元后缀
     */
    SuffixBuildingUnit,
    /**
     * 数字号后缀
     */
    SuffixNumber,
    /**
     * 开始状态
     */
    Start,
    /**
     * 结束状态
     */
    End,
    /**
     * (
     */
    StartSuffix,
    /**
     * )
     */
    EndSuffix,
    /**
     * 不确定语素
     */
    Unknow,
    /**
     * 非地址语素
     */
    Other,
    /**
     * 省后缀
     */
    SuffixProvince,
    /**
     * 市后缀
     */
    SuffixCity,
    /**
     * 区后缀
     */
    SuffixCounty,
    /**
     * 区域
     */
    District,
    /**
     * 区域后缀
     */
    SuffixDistrict,
    /**
     * 镇后缀
     */
    SuffixTown,
    /**
     * 街后缀
     */
    SuffixStreet,
    /**
     * 地标建筑后缀
     */
    SuffixLandMark,
    /**
     * 村后缀
     */
    SuffixVillage,
    /**
     * 指示性设施后缀
     */
    SuffixIndicationFacility,
    /**
     * 指示性设施
     */
    IndicationFacility,
    /**
     * 指示性设施方位后缀
     */
    SuffixIndicationPosition,
    /**
     * 指示性设施方位
     */
    IndicationPosition,
    /**
     * 连接词
     */
    Conj,
}
