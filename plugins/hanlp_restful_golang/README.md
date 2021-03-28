# gohanlp
[HanLP](https://github.com/hankcs/HanLP) The multilingual NLP library for researchers and companies, built on PyTorch and TensorFlow 2.x, for advancing state-of-the-art deep learning techniques in both academia and industry. HanLP was designed from day one to be efficient, user friendly and extendable. It comes with pretrained models for various human languages including English, Chinese and many others.



## Usage

### install

```
go get -u github.com/xxjwxc/gohanlp@master

```

#### Apply for auth certification

https://bbs.hanlp.com/t/hanlp2-1-restful-api/53

#### text model

```
client := hanlp.HanLPClient(hanlp.WithAuth("The auth you applied for")) // If not, anonymity
s, _ := client.Parse("In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.",hanlp.WithLanguage("mul"))
fmt.Println(s)
```

#### object model

```
client := hanlp.HanLPClient(hanlp.WithAuth("The auth you applied for")) // If not, anonymity
resp, _ := client.ParseObj("In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.",hanlp.WithLanguage("mul"))
fmt.Println(resp)
```


