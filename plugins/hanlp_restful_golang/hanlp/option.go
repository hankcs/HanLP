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

// WithURL set hanlp address
func WithURL(url string) Option {
	return func(o *Options) {
		o.URL = url
	}
}

// WithAuth set auth
func WithAuth(auth string) Option {
	return func(o *Options) {
		o.Auth = auth
	}
}

// WithLanguage set language
func WithLanguage(language string) Option {
	return func(o *Options) {
		o.Language = language
	}
}

// WithTimeout set timeout
func WithTimeout(timeout time.Time) Option {
	return func(o *Options) {
		o.Timeout = timeout
	}
}

// WithTasks set tasks list("tok","ud","ner","srl","sdp/dm","sdp/pas","sdp/psd","con")
func WithTasks(tasks ...string) Option {
	return func(o *Options) {
		o.Tasks = append(o.Tasks, tasks...)
	}
}

// WithSkipTasks set skip tasks list("tok","ud","ner","srl","sdp/dm","sdp/pas","sdp/psd","con")
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
