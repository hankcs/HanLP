# Golang RESTful API

## Install

```shell script
go get -u github.com/hankcs/gohanlp@main
```

## Quick Start 

Obtain an `auth` from any compatible service provider like our [free service](https://bbs.hankcs.com/t/apply-for-free-hanlp-restful-apis/3178), then initiate a `HanLPClient` and call its `Parse` interface.

```java
package main

import (
	"fmt"
	"github.com/hankcs/gohanlp/hanlp"
)

func main() {
    client := hanlp.HanLPClient(hanlp.WithAuth("The auth you applied for")) // anonymous users can skip auth
    s, _ := client.Parse("In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.",hanlp.WithLanguage("mul"))
    fmt.Println(s)
}
```

Refer to our [testcases](https://github.com/hankcs/gohanlp/blob/main/main_test.go) and [data format](../data_format) for more details.

