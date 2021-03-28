package gohanlp

import (
	"fmt"
	"testing"

	"github.com/xxjwxc/gohanlp/hanlp"
)

func TestMain(t *testing.T) {
	client := hanlp.HanLPClient(hanlp.WithAuth("")) // auth

	s, _ := client.Parse("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
		hanlp.WithLanguage("zh"))
	fmt.Println(s)

	resp, _ := client.ParseObj("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
		hanlp.WithLanguage("zh"))
	fmt.Println(resp)
}
