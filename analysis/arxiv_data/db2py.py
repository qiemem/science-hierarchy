import sqlite3

def get_docs(dbfilenamew, subject="cs"):
    """
    Gets all documents as (title, body) pairs for the given subject. Possible
    subjects are cs, math, physics, stat, q-bio, q-fin, nlin, gr-qc, cond-mat,
    astro-ph...
    """
    with sqlite3.connect(dbfilenamew) as conn:
        conn.text_factory = str
        c = conn.cursor()
        query = "SELECT title, abstract FROM records WHERE categories LIKE '{0}%' OR categories LIKE '% {0}.%';"
        return list(c.execute(query.format(subject)))

