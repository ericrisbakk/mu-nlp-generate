import math, collections
from audioop import avg
from builtins import print
from random import randint
from random import random

import pandas as pd


def start():
    """Initialize your data structures in the constructor."""
    # TODO your code here

    gramNumber = 2  # N-gram
    lyricsSize = 15

    print("test")
    corpus = pd.read_csv("ProcessedSongData.csv")
    data = corpus.corrected


    avgLyricsLength = 0; # how many words do i need

    for i in range(len(data)):
        avgLyricsLength += len(data[i])
    avgLyricsLength = round(round(avgLyricsLength / len(data),0))

    countsOfList = []
    for i in range(gramNumber):
        countsOfList.append(collections.defaultdict(lambda: 0))

    bag = tokenize(data)
    #generate word counts
    for j in range(gramNumber):
        for i in range(len(data)):
            words = []
            for p in range(j):
                words = words + ["<s>"]
            words = words + data[i].split()

            for l in range(j, len(words)):
                token = words[l]
                for k in range(1,j+1):
                    token = words[l - k] + " " + token
                countsOfList[j][token] = countsOfList[j][token] + 1

    #generate lyrics
    lyrics = []
    for i in range(gramNumber - 1):
        lyrics.append("<s>")

    if(len(lyrics) < 1):
        r = randint(0, len(bag)-1)
        lyrics.append(bag[r])

    while len(lyrics) < lyricsSize:
        toMatch = lyrics[len(lyrics)-1]

        found = False
        for i in range(gramNumber):
            if(i >= gramNumber -1):
                r = randint(0, len(bag) - 1)
                lyrics.append(bag[r])

            for t in range(len(lyrics) - 1 - (gramNumber - i - 2), len(lyrics) - 1):
                toMatch = lyrics[t] + " " + toMatch

            toAdd = ""
            bestScore = 0
            for key in countsOfList[gramNumber - i -1]:
                newKey = ""
                keyAsList = key.split()
                newKey += keyAsList[0]
                for q in range(1,len(keyAsList)-1):
                    newKey = newKey + " " + keyAsList[q]

                if toMatch == newKey:
                    if(randint(1,10) > 9):
                        bestScore = countsOfList[gramNumber - i - 1].get(key)
                        toAdd = keyAsList[len(keyAsList) - 1]
                    if countsOfList[gramNumber - i -1].get(key) > bestScore:
                        bestScore = countsOfList[gramNumber - i-1].get(key)
                        toAdd = keyAsList[len(keyAsList)-1]

            if(not found and bestScore > 0):
                found = True
                lyrics.append(toAdd)
                i = gramNumber + 1

            if found:
                break

        print(lyrics)





def tokenize(data):
    bag = []
    for i in range(len(data)):

        words = data[i].split()
        for w in words:
            bag.append(w)

    bag = sorted(list(set(bag)))
    return bag

start()
