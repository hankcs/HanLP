package hanlp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"

	"github.com/xxjwxc/public/myhttp"
	"github.com/xxjwxc/public/mylog"
	"github.com/xxjwxc/public/tools"
)

type hanlp struct {
	opts Options
}

// HanLPClient build client
func HanLPClient(opts ...Option) *hanlp {
	options := Options{ // default
		URL:      "https://www.hanlp.com/api",
		Language: "zh",
	}

	for _, f := range opts { // deal option
		f(&options)
	}

	return &hanlp{
		opts: options,
	}
}

// Parse deal
func (h *hanlp) Parse(text string, opts ...Option) (string, error) {
	options := h.opts
	for _, f := range opts { // option
		f(&options)
	}

	req := &HanReq{
		Text:      text,
		Language:  options.Language, // (zh,mnl)
		Tasks:     options.Tasks,
		SkipTasks: options.SkipTasks,
	}
	b, err := myhttp.PostHeader(options.URL+"/parse", tools.JSONDecode(req), getHeader(options))
	if err != nil {
		mylog.Error(err)
		return "", err
	}

	return string(b), nil
}

// Parse parse object
func (h *hanlp) ParseObj(text string, opts ...Option) (*HanResp, error) {
	options := h.opts
	for _, f := range opts { // option
		f(&options)
	}

	req := &HanReq{
		Text:      text,
		Language:  options.Language, // (zh,mnl)
		Tasks:     options.Tasks,
		SkipTasks: options.SkipTasks,
	}
	b, err := myhttp.PostHeader(options.URL+"/parse", tools.JSONDecode(req), getHeader(options))
	if err != nil {
		mylog.Error(err)
		return nil, err
	}

	return marshalHanResp(b)
}

// ParseAny parse any request parms
func (h *hanlp) ParseAny(text string, resp interface{}, opts ...Option) error {
	reqType := reflect.TypeOf(resp)
	if reqType.Kind() != reflect.Ptr {
		return fmt.Errorf("req type not a pointer:%v", reqType)
	}

	options := h.opts
	for _, f := range opts { // option
		f(&options)
	}

	req := &HanReq{
		Text:      text,
		Language:  options.Language, // (zh,mnl)
		Tasks:     options.Tasks,
		SkipTasks: options.SkipTasks,
	}
	b, err := myhttp.PostHeader(options.URL+"/parse", tools.JSONDecode(req), getHeader(options))
	if err != nil {
		mylog.Error(err)
		return err
	}

	switch v := resp.(type) {
	case *string:
		*v = string(b)
	case *[]byte:
		*v = b
	case *HanResp:
		tmp, e := marshalHanResp(b)
		*v, err = *tmp, e
	default:
		err = json.Unmarshal(b, v)
	}

	if err != nil {
		return err
	}

	return nil
}

// marshal obj
func marshalHanResp(b []byte) (*HanResp, error) {
	var hr hanResp
	err := json.Unmarshal(b, &hr)
	if err != nil {
		mylog.Error(err)
		return nil, err
	}
	resp := &HanResp{
		TokFine:   hr.TokFine,
		TokCoarse: hr.TokCoarse,
		PosCtb:    hr.PosCtb,
		PosPku:    hr.PosPku,
		Pos863:    hr.Pos863,
	}

	// ner/pku
	for _, v := range hr.NerPku {
		var tmp []NerTuple
		for _, v1 := range v {
			switch t := v1.(type) {
			case []interface{}:
				{
					tmp = append(tmp, NerTuple{
						Entity: t[0].(string),
						Type:   t[1].(string),
						Begin:  int(t[2].(float64)),
						End:    int(t[3].(float64)),
					})
				}
			default:
				mylog.Error("%v : not unmarshal", t)
			}
		}
		resp.NerPku = append(resp.NerPku, tmp)
	}
	// ----------end

	// ner/msra
	for _, v := range hr.NerMsra {
		var tmp []NerTuple
		for _, v1 := range v {
			switch t := v1.(type) {
			case []interface{}:
				{
					tmp = append(tmp, NerTuple{
						Entity: t[0].(string),
						Type:   t[1].(string),
						Begin:  int(t[2].(float64)),
						End:    int(t[3].(float64)),
					})
				}
			default:
				mylog.Error("%v : not unmarshal", t)
			}
		}
		resp.NerMsra = append(resp.NerMsra, tmp)
	}
	// ----------end

	// ner/ontonotes
	for _, v := range hr.NerOntonotes {
		var tmp []NerTuple
		for _, v1 := range v {
			switch t := v1.(type) {
			case []interface{}:
				{
					tmp = append(tmp, NerTuple{
						Entity: t[0].(string),
						Type:   t[1].(string),
						Begin:  int(t[2].(float64)),
						End:    int(t[3].(float64)),
					})
				}
			default:
				mylog.Error("%v : not unmarshal", t)
			}
		}
		resp.NerOntonotes = append(resp.NerOntonotes, tmp)
	}
	// ----------end

	// srl
	for _, v := range hr.Srl {
		var tmp [][]SrlTuple
		for _, v1 := range v {
			var tmp1 []SrlTuple
			for _, v2 := range v1 {
				switch t := v2.(type) {
				case []interface{}:
					{
						tmp1 = append(tmp1, SrlTuple{
							ArgPred: t[0].(string),
							Label:   t[1].(string),
							Begin:   int(t[2].(float64)),
							End:     int(t[3].(float64)),
						})
					}
				default:
					mylog.Error("%v : not unmarshal", t)
				}
			}
			tmp = append(tmp, tmp1)
		}
		resp.Srl = append(resp.Srl, tmp)
	}
	// -------------end

	// dep
	for _, v := range hr.Dep {
		var tmp []DepTuple
		for _, v1 := range v {
			switch t := v1.(type) {
			case []interface{}:
				{
					tmp = append(tmp, DepTuple{
						Head:     int(t[0].(float64)),
						Relation: t[1].(string),
					})
				}
			default:
				mylog.Error("%v : not unmarshal", t)
			}
		}
		resp.Dep = append(resp.Dep, tmp)
	}
	// ------------end
	// sdp
	for _, v := range hr.Sdp {
		var tmp [][]DepTuple
		for _, v1 := range v {
			var tmp1 []DepTuple
			for _, v2 := range v1 {
				switch t := v2.(type) {
				case []interface{}:
					{
						tmp1 = append(tmp1, DepTuple{
							Head:     int(t[0].(float64)),
							Relation: t[1].(string),
						})
					}
				default:
					mylog.Error("%v : not unmarshal", t)
				}
			}
			tmp = append(tmp, tmp1)
		}
		resp.Sdp = append(resp.Sdp, tmp)
	}
	// ------------end
	// Con
	resp.Con = dealCon(hr.Con)
	// ------------end

	// Con          []interface{}
	return resp, nil
}

func getHeader(opts Options) http.Header {
	header := make(http.Header)
	header.Add("Accept", "application/json")
	header.Add("Content-Type", "application/json;charset=utf-8")
	if len(opts.Auth) > 0 {
		header.Add("Authorization", "Basic "+opts.Auth)
	}
	return header
}

func dealCon(info []interface{}) (re []ConTuple) {
	if len(info) == 0 {
		return nil
	}

	switch t := info[0].(type) {
	case string:
		{
			tmp1 := ConTuple{
				Key: t,
			}
			if len(info) == 2 {
				tmp1.Value = dealCon(info[1].([]interface{}))
			}
			// else { // It doesn't exist in theory
			// 	fmt.Println(info)
			// }
			re = append(re, tmp1)
		}
	case []interface{}:
		{
			for _, t1 := range info {
				tmp1 := ConTuple{}
				tmp1.Value = dealCon(t1.([]interface{}))
				re = append(re, tmp1)
			}
		}
	}

	return re
}
