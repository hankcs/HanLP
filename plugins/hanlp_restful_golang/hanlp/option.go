package hanlp

import (
	"time"
)

// Options opts define
type Options struct {
	URL       string
	Auth      string
	Language  string
	Timeout   time.Time
	Tasks     []string
	SkipTasks []string
	OutPut    interface{}
	Tokens    []string
}

// Option opts list func
type Option func(*Options)

// WithURL 设置hanlp地址
func WithURL(url string) Option {
	return func(o *Options) {
		o.URL = url
	}
}

// WithAuth 设置授权码
func WithAuth(auth string) Option {
	return func(o *Options) {
		o.Auth = auth
	}
}

// WithLanguage 设置语言
func WithLanguage(language string) Option {
	return func(o *Options) {
		o.Language = language
	}
}

// WithTimeout 调用超时设置
func WithTimeout(timeout time.Time) Option {
	return func(o *Options) {
		o.Timeout = timeout
	}
}

// WithTasks 设置任务列表("tok","ud","ner","srl","sdp/dm","sdp/pas","sdp/psd","con")
func WithTasks(tasks ...string) Option {
	return func(o *Options) {
		o.Tasks = append(o.Tasks, tasks...)
	}
}

// WithSkipTasks 设置忽略的任务列表("tok","ud","ner","srl","sdp/dm","sdp/pas","sdp/psd","con")
func WithSkipTasks(skipTasks ...string) Option {
	return func(o *Options) {
		o.SkipTasks = append(o.SkipTasks, skipTasks...)
	}
}

func WithTokens(tokens ...string) Option {
	return func(o *Options) {
		o.Tokens = append(o.Tokens, tokens...)
	}
}
