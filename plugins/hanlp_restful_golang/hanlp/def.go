package hanlp

// https://hanlp.hankcs.com/docs/data_format.html

// HanReq hanlp 请求参数
type HanReq struct {
	Text      string   `json:"text,omitempty"`     // 文本
	Language  string   `json:"language,omitempty"` // 语言(zh,mnt)
	Tokens    []string `json:"tokens,omitempty"`   // 一个句子列表，其中每个句子都是一个标记列表。
	Tasks     []string `json:"tasks,omitempty"`    // 任务列表()
	SkipTasks []string `json:"skip_tasks"`
}

// HanResp hanlp 返回参数
type HanResp struct {
	TokFine      [][]string     `json:"tok/fine"`      // 分词 细粒度的标记化模型
	TokCoarse    [][]string     `json:"tok/coarse"`    // 分词 粗粒度的标记化模型
	PosCtb       [][]string     `json:"pos/ctb"`       // 词性标注 https://hanlp.hankcs.com/docs/annotations/pos/ctb.html
	PosPku       [][]string     `json:"pos/pku"`       // 词性标注 https://hanlp.hankcs.com/docs/annotations/pos/pku.html
	Pos863       [][]string     `json:"pos/863"`       // 词性标注 https://hanlp.hankcs.com/docs/annotations/pos/863.html
	NerPku       [][]NerTuple   `json:"ner/pku"`       // 命名实体识别 https://hanlp.hankcs.com/docs/annotations/ner/pku.html
	NerMsra      [][]NerTuple   `json:"ner/msra"`      // 命名实体识别 https://hanlp.hankcs.com/docs/annotations/ner/msra.html
	NerOntonotes [][]NerTuple   `json:"ner/ontonotes"` // 命名实体识别 https://hanlp.hankcs.com/docs/annotations/ner/ontonotes.html
	Srl          [][][]SrlTuple `json:"srl"`           // 语义角色标注 其中谓词被标记为pred https://hanlp.hankcs.com/docs/annotations/srl/index.html
	Dep          [][]DepTuple   `json:"dep"`           // 依存句法分析 https://hanlp.hankcs.com/docs/annotations/dep/index.html
	Sdp          [][][]DepTuple `json:"sdp"`           // 语义依存分析 https://hanlp.hankcs.com/docs/annotations/sdp/index.html
	Con          []ConTuple     `json:"con"`           // 短语成分分析
}

// NerTuple
type NerTuple struct {
	Entity string `json:"entity"` // 实体
	Type   string `json:"type"`   // 类型
	Begin  int    `json:"begin"`  // 开始点
	End    int    `json:"end"`    // 独占偏移量
}

// SrlTuple 语义角色标注。与ner类似，每个元素都是一个元组（arg/pred，label，begin，end），其中谓词被标记为pred。
type SrlTuple struct {
	ArgPred string `json:"arg/pred"` // 实体
	Label   string `json:"label"`    // 类型
	Begin   int    `json:"begin"`    // 开始点
	End     int    `json:"end"`      // 独占偏移量
}

// DepTuple 依赖关系分析。每个元素都是（head，relationship）的元组，其中head以索引0（它是根）开头。
type DepTuple struct {
	Head     int    `json:"head"`     // 以索引0（它是根）开头
	Relation string `json:"relation"` // 关系
}

// ConTuple 每个列表都是一个括号内的组成部分
type ConTuple struct {
	Key   string     `json:"key"`   // 成分
	Value []ConTuple `json:"value"` // 如果有
}

type hanResp struct {
	TokFine      [][]string        `json:"tok/fine"`      // 分词 细粒度的标记化模型
	TokCoarse    [][]string        `json:"tok/coarse"`    // 分词 粗粒度的标记化模型
	PosCtb       [][]string        `json:"pos/ctb"`       // 词性标注 https://hanlp.hankcs.com/docs/annotations/pos/ctb.html
	PosPku       [][]string        `json:"pos/pku"`       // 词性标注 https://hanlp.hankcs.com/docs/annotations/pos/pku.html
	Pos863       [][]string        `json:"pos/863"`       // 词性标注 https://hanlp.hankcs.com/docs/annotations/pos/863.html
	NerPku       [][]interface{}   `json:"ner/pku"`       // 命名实体识别 https://hanlp.hankcs.com/docs/annotations/ner/pku.html
	NerMsra      [][]interface{}   `json:"ner/msra"`      // 命名实体识别 https://hanlp.hankcs.com/docs/annotations/ner/msra.html
	NerOntonotes [][]interface{}   `json:"ner/ontonotes"` // 命名实体识别 https://hanlp.hankcs.com/docs/annotations/ner/ontonotes.html
	Srl          [][][]interface{} `json:"srl"`           // 语义角色标注 其中谓词被标记为pred https://hanlp.hankcs.com/docs/annotations/srl/index.html
	Dep          [][]interface{}   `json:"dep"`           // 依存句法分析 https://hanlp.hankcs.com/docs/annotations/dep/index.html
	Sdp          [][][]interface{} `json:"sdp"`           // 语义依存分析 https://hanlp.hankcs.com/docs/annotations/sdp/index.html
	Con          []interface{}     `json:"con"`           // 短语成分分析
}
