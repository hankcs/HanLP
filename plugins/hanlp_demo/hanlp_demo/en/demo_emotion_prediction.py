# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-01 03:52
import hanlp

classifier = hanlp.load('EMPATHETIC_DIALOGUES_SITUATION_ALBERT_BASE_EN')
print(classifier('I received concert tickets for Christmas.'))
print(classifier(["My wife got a new job with Google.",
                  "Do you like to go through old family pictures?",
                  "Thank God it's friday"]))
