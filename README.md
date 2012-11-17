The data is stored in `data/neuro.sqlite`. It's a sqlite3 database, which is just a file; no server needed. But you will need sqlite3 to interact with it. I believe it comes in OS X, but if not, just use homebrew or something to get it.

`data/db2py.py` is a python script with utility functions to dump the abstracts in a python list. You probably want the `get_clean_docs()` function, which strips out all non-letters and non-spaces.
