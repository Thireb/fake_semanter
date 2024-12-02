"""This module is to host stopwords"""
import string

# Get roman-urdu stopwords
# This stopwords have been taken from the git repository
# https://github.com/haseebelahi/roman-urdu-stopwords/blob/master/stopwords.txt
# These words have not been verified
STOPWORDS = ['ai', 'ayi', 'hy', 'hai', 'main', 'ki', 'tha', 'koi', 'ko', 'sy',
             'woh', 'bhi', 'aur', 'wo', 'yeh', 'rha', 'hota', 'ho', 'ga', 'ka',
             'le', 'lye', 'kr', 'kar', 'lye', 'liye', 'hotay', 'waisay', 'gya',
             'gaya', 'kch', 'ab', 'thy', 'thay', 'houn', 'hain', 'han', 'to',
             'is', 'hi', 'jo', 'kya', 'thi', 'se', 'pe', 'phr', 'wala', 'waisay',
             'us', 'na', 'ny', 'hun', 'rha', 'raha', 'ja', 'rahay', 'abi',
             'uski', 'ne', 'haan', 'acha', 'nai', 'sent', 'photo', 'you', 'kafi',
             'gai', 'rhy', 'kuch', 'jata', 'aye', 'ya', 'dono', 'hoa', 'aese',
             'de', 'wohi', 'jati', 'jb', 'krta', 'lg', 'rahi', 'hui', 'karna',
             'krna', 'gi', 'hova', 'yehi', 'jana', 'jye', 'chal', 'mil', 'tu',
             'hum', 'par', 'hay', 'kis', 'sb', 'gy', 'dain', 'krny', 'tou']

# Add punctuation to stopwords list
STOPWORDS += list(string.punctuation)
STOPWORDS += ["''", '""', '...', '``']
