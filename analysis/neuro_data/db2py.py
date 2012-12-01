import sqlite3
import re

def get_doc_texts(dbfilename):
    # Gets a list of all (title, abstract) pairs from the database.
    with sqlite3.connect(dbfilename) as conn:
        # see:
        # http://bytes.com/topic/python/answers/703569-sqlite3-operationalerror-could-not-decode-utf-8-column
        conn.text_factory = str
        c = conn.cursor()
        return list(c.execute('SELECT title, abstract FROM wos_document;'))
