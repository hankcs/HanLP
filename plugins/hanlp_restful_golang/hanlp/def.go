package hanlp

// https://hanlp.hankcs.com/docs/data_format.html

// HanReq hanlp
type HanReq struct {
	Text      string   `json:"text,omitempty"`
	Language  string   `json:"language,omitempty"` // (zh,mnl)
	Tokens    []string `json:"tokens,omitempty"`
	Tasks     []string `json:"tasks,omitempty"`
	SkipTasks []string `json:"skip_tasks"`
}

// HanResp hanlp 返回参数
type HanResp struct {
	TokFine      [][]string     `json:"tok/fine"`
	TokCoarse    [][]string     `json:"tok/coarse"`
	PosCtb       [][]string     `json:"pos/ctb"`
	PosPku       [][]string     `json:"pos/pku"`
	Pos863       [][]string     `json:"pos/863"`
	NerPku       [][]NerTuple   `json:"ner/pku"`
	NerMsra      [][]NerTuple   `json:"ner/msra"`
	NerOntonotes [][]NerTuple   `json:"ner/ontonotes"`
	Srl          [][][]SrlTuple `json:"srl"`
	Dep          [][]DepTuple   `json:"dep"`
	Sdp          [][][]DepTuple `json:"sdp"`
	Con          []ConTuple     `json:"con"`
}

// NerTuple
type NerTuple struct {
	Entity string `json:"entity"`
	Type   string `json:"type"`
	Begin  int    `json:"begin"`
	End    int    `json:"end"`
}

// SrlTuple
type SrlTuple struct {
	ArgPred string `json:"arg/pred"`
	Label   string `json:"label"`
	Begin   int    `json:"begin"`
	End     int    `json:"end"`
}

// DepTuple
type DepTuple struct {
	Head     int    `json:"head"`
	Relation string `json:"relation"`
}

// ConTuple
type ConTuple struct {
	Key   string     `json:"key"`
	Value []ConTuple `json:"value"`
}

type hanResp struct {
	TokFine      [][]string        `json:"tok/fine"`
	TokCoarse    [][]string        `json:"tok/coarse"`
	PosCtb       [][]string        `json:"pos/ctb"`       //  https://hanlp.hankcs.com/docs/annotations/pos/ctb.html
	PosPku       [][]string        `json:"pos/pku"`       //  https://hanlp.hankcs.com/docs/annotations/pos/pku.html
	Pos863       [][]string        `json:"pos/863"`       //  https://hanlp.hankcs.com/docs/annotations/pos/863.html
	NerPku       [][]interface{}   `json:"ner/pku"`       //  https://hanlp.hankcs.com/docs/annotations/ner/pku.html
	NerMsra      [][]interface{}   `json:"ner/msra"`      //  https://hanlp.hankcs.com/docs/annotations/ner/msra.html
	NerOntonotes [][]interface{}   `json:"ner/ontonotes"` //  https://hanlp.hankcs.com/docs/annotations/ner/ontonotes.html
	Srl          [][][]interface{} `json:"srl"`           //  https://hanlp.hankcs.com/docs/annotations/srl/index.html
	Dep          [][]interface{}   `json:"dep"`           //  https://hanlp.hankcs.com/docs/annotations/dep/index.html
	Sdp          [][][]interface{} `json:"sdp"`           //  https://hanlp.hankcs.com/docs/annotations/sdp/index.html
	Con          []interface{}     `json:"con"`           //
}
