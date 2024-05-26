# Local-GenAI Search - Local generative search

Local GenAI Search is your local generative search engine 
based on Llama3 8B model that can run localy on 32GB 
laptop or computer (developed with MacBookPro M2 with 32BG RAM)

## How to run

In order to run your Local Generative AI Search (given you have sufficiently string machine to run Llama3), you need to 
download the repository:

````
git clone https://github.com/nikolamilosevic86/local-gen-search.git
````
You will need to install all the requirements:
```commandline
pip install -r requirements.txt
```

The next step is to index a folder and its subfolders containing
documents that you would like to search. You can do it using
the ``index.py`` file. Run

```commandline
python index.py path/to/folder
```
This will create a qdrant client index locally and index all the files
in this folder and its subfolders with extensions ```.pdf```

The next step would be to run the generative search service.
For this you can run:

```commandline
python uvicorn_start.py
```