# Java RESTful API

Add the following dependency into the `pom.xml` file of your project. 

```xml
<dependency>
  <groupId>com.hankcs.hanlp.restful</groupId>
  <artifactId>hanlp-restful</artifactId>
  <version>0.0.3</version>
</dependency>
```

Obtain an `auth` from any compatible service provider like our [free service](https://bbs.hankcs.com/t/apply-for-free-hanlp-restful-apis/3178), then initiate a `HanLPClient` and call its `parse` interface.

```java
HanLPClient client = new HanLPClient("https://hanlp.hankcs.com/api", null); // Replace null with your auth
System.out.println(client.parse("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。英首相与特朗普通电话讨论华为与苹果公司。"));
```

Refer to our [testcases](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_restful_java/src/test/java/com/hankcs/hanlp/restful/HanLPClientTest.java) and [data format](../data_format) for more details.

