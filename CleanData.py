# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:09:10 2019

This file loads the original dato, performs some preprocessing steps
on it, and then saves a new csv file with this new information,
for later, easier use.

@author: heier
"""

# Imports
import pandas as pd
import re
from spellchecker import SpellChecker

# Not used, currently.
def add_start_and_end_symbols(sentence):
    return ("<start> " + sentence + " <stop>")

def basic_cleaning(sentence):
    s = sentence.lower()
    s = s.replace("\n", " \n ")
    return s
   # The below removes anything except whitespace and alphanumeric characters.
def remove_punctuation(s):
    return re.sub('\\.|[^\w\s]', ' ', s)

# Turn sentence into list of words.
def tokenize(s):
    s_list = [w for w in s.split(' ') if w.strip() != '' or w == '\n']      
    return s_list

# There are, to my awareness, no words with consecutive 
# three same letters in english.
def remove_extra_letters(s):
    return re.sub(r"(.)\1{2,}", r"\1"*2, s)   

# Replaces unknown words in list according to u_dict (given later)
def replace_unknown(s_list):
    for i, s in enumerate(s_list):
        if s in u_dict:
            s_list[i] = u_dict[s]
    return s_list

# Load data.
data = pd.read_csv("LocalData/SongData.csv")
print("Loaded dataset.")
# Methods for preprocessing. These methods
# are applied to the entire sentence. 

print("Starting to clean and tokenize.")
# Clean the data
data['clean'] = data['text'].apply(basic_cleaning)
print("Cleaning applied.")
data['clean'] = data['clean'].apply(remove_punctuation)
print("Punctuation removed.")
data['clean'] = data['clean'].apply(remove_extra_letters)
print("Extra letters removed.")
token_data = data['clean'].apply(tokenize)
print("Strings tokenized.")

print("\n", repr(data.text[0]), "\n")
print("\n", repr(data.text[0][30:50]), "\n")
print("\n", repr(data.clean[0]), "\n")
print("\n", repr(data.clean[0][30:50]), "\n")


# get unique tokens.
print("Getting unique tokens.")
text = token_data.values
tokens = []
for song in text:
    tokens.extend(song)

tokens = list(set(tokens))

# Find all unknown words.
spell = SpellChecker()
spell.distance = 1

# Some people like to say individual letters. adding them
# Adding to spell checker.
a = ord('a') 
individual_letters = [chr(i) for i in range(a,a+26)]

spell.word_frequency.load_words(individual_letters)
spell.word_frequency.load_words(["\n"])

print("Getting unrecognized tokens (relative to English)")
unknown = spell.unknown(tokens)

# Add unknown word and guessed correction.
# Not added if the word is the same.
# to be made into a dictionary later
u_list = []
for u in unknown:
    correct = ((u, spell.correction(u)))
    if correct[0] != correct[1]:
        u_list.append(correct)
        
# Remove possible duplicates
u_list = list(set(u_list))
u_dict = dict(u_list)

print("Creating column of 'corrected' spellings")
corrected_data = token_data.apply(replace_unknown)

print("converted tokenized corrected spellings back to string")
def convert_s_list_to_string(s_list):
    return " ".join(s_list)

data['corrected'] = corrected_data.apply(convert_s_list_to_string)

print("\n", repr(data.corrected[0]))

print("\nSaving CSV file")
data.to_csv("LocalData/ProcessedSongData.csv")
print("CSV file saved.")










    















# End.