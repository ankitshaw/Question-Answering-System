from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.file_converter.pdf import PDFToTextConverter
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline

from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from flask import Flask, request, jsonify

from werkzeug.utils import secure_filename

import os
import json
import logging


def preprocessing(path):
    directory = path
    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["de","en"])
    processor = PreProcessor(clean_empty_lines=True,
                            clean_whitespace=True,
                            clean_header_footer=True,
                            split_by="word",
                            split_length=200,
                            split_respect_sentence_boundary=True)
    docs = []
    for filename in os.listdir(directory):
        d = converter.convert(os.path.join(directory, filename), meta={"name":filename})
        d = processor.process(d)
        docs.extend(d)

    # Let's have a look at the first 3 entries:
    print(docs[:3])
    return docs

def retriever(document_store):
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_query=64,
                                      max_seq_len_passage=256,
                                      batch_size=2,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True
                                      )

    return retriever


def main_test():
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    
    docs = preprocessing("data")
    document_store.write_documents(docs)
    
    retriever = retriever(document_store)
    document_store.update_embeddings(retriever)
    
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    
    pipe = ExtractiveQAPipeline(reader, retriever)
    
    prediction = pipe.run(query="Who is a counterparty?", top_k_retriever=5, top_k_reader=5)
    print_answers(prediction, details="minimal")


@app.route('/query',methods=['GET', 'POST'])
def search():
    """Return the n answers."""

    question = request.get_json()
    question = question['questions']

    prediction = pipe.run(query=question[0], top_k_retriever=3, top_k_reader=3)
    answer = []
  
    for res in prediction['answers']:
        answer.append(res['answer'])

    result = {"results":[prediction]}
    return json.dumps(result)

@app.route('/file-upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'File Uploaded Successfully'


def main_api():
    app = Flask(__name__)
    CORS(app)
    run_with_ngrok(app)
    app.run()


