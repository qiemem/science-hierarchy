import sqlite3
import re

def get_doc_texts():
    # Gets a list of all (title, abstract) pairs from the database.
    with sqlite3.connect('neuro.sqlite') as conn:
        # see:
        # http://bytes.com/topic/python/answers/703569-sqlite3-operationalerror-could-not-decode-utf-8-column
        conn.text_factory = str
        c = conn.cursor()
        return list(c.execute('SELECT title, abstract FROM wos_document;'))

def strip_non_alphanum(string, pattern=re.compile('[^A-Za-z ]')):
    # Strips the given string of all non-letters and non-spaces
    return pattern.sub('', string)

def get_clean_docs():
    # Gets a list of the concatenated titles and abstracts with all non-letter
    # and non-spaces removed.
    return [strip_non_alphanum(title+' '+text).lower() for (title, text) in get_doc_texts()]
