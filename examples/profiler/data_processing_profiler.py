# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import List, Tuple
import yaml
from termcolor import colored
import texar.torch as tx


from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from forte.processors.nlp import CoNLLNERPredictor
from ft.onto.base_ontology import Sentence, Document, Token, EntityMention
from fortex.nltk import NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter


class DummyPackProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        pass

class OntonoteDataProcessing():
    def __init__(self, file_path = "data_samples/profiler/combine_data"):
        self.root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
            )
        )
        self.dataset_path = os.path.join(
            self.root_path, file_path
        )
        # Define and config the Pipeline
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(OntonotesReader())
        # self.nlp.add(DummyPackProcessor())
        self.nlp.add(NLTKSentenceSegmenter())
        self.nlp.add(NLTKWordTokenizer())
        self.nlp.add(NLTKPOSTagger())
        
        # config = yaml.safe_load(open(os.path.join(root_path, "examples/profiler/config.yml"), "r"))
        # config = Config(config, default_hparams=None)
        # self.nlp.add(CoNLLNERPredictor(), config=config.NER)

        self.nlp.set_profiling(True)

        self.nlp.initialize()

        self.tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
    
    def whole_data_processing(self):
        file_path = "data_samples/profiler/whole_data"
        dataset_path = os.path.join(
            self.root_path, file_path
        )
        tokenize_time = 0
        get_document_time = 0
        get_sentence_time = 0
        get_token_entity_time = 0
        get_data_time = 0
        iter_time = 0
        get_token_entity_from_document_time = 0
        # get processed pack from dataset
        bg = time.time()
        iter = self.nlp.process_dataset(dataset_path)
        # print("length of iterator: ", len(iter))
        process_time = time.time() - bg

        """
        iter_time:
        Iterate through the dataset and get each datapack
        """
        it = time.time()
        packcnt = 0
        sentcnt = 0
        token_sent_cnt = 0
        token_doc_cnt = 0
        for pack in iter:
            packcnt += 1
            iter_time += time.time() - it
            
            """
                get_document_time:
                Get document from databack (expected to have only one document each pack)
            """
            dt = time.time()
            document = []
            for doc in pack.get(Document):
                document.append(doc)
            if len(document) > 1:
                raise RuntimeError("More than one document in a datatuple")
            document = document[0]
            document_text = document.text
            get_document_time += time.time() - dt

            """
                get_sentence_time:
                Get sentence from databack
            """
            st = time.time()
            sentences = []
            for sent in pack.get(Sentence):
                sentcnt += 1
                sent_text = sent.text
                sentences.append(sent_text)
                get_sentence_time += time.time()-st
                
                tet = time.time()
                tokenized_tokens = []
                tokens = [t.text for t in pack.get(Token, sent)]
                entities = [entity.text
                    for entity in pack.get(EntityMention, sent)
                ]
                get_token_entity_time += time.time() - tet

                """
                tokenize_time: 
                Tokenize tokens and entities with BERTTokenizer
                """
                tt = time.time()
                tokenized_tokens += self.tokenizer.map_text_to_id(sent_text)
                token_ids = self.tokenizer.map_token_to_id(tokens)
                entity_ids = self.tokenizer.map_token_to_id(entities)
                tokenize_time += time.time()-tt

                st = time.time()
            
            """
            get_token_entity_from_document_time:
            Get token and entity from the whole document
            """
            tokdocTime = time.time()
            tokenized_tokens = []
            tokens = [t.text for t in pack.get(Token, document)]
            token_doc_cnt += len(tokens)
            entities = [entity.text
                for entity in pack.get(EntityMention, document)
            ]
            get_token_entity_from_document_time += time.time() - tokdocTime

            
            """
            get_data_time:
            Get sentence attributes with get_data request
            """
            datat = time.time()
            sentences = []
            request = {
                Sentence: ["speaker", "part_id"]
            }
            for sent in pack.get_data(Sentence, request):
                sentences.append(sent['context'])
            
            sentences_text_comp = " ".join(sentences)
            # assert(sentences_text == sentences_text_comp)
            get_data_time += time.time() - datat
            it = time.time()
        
        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_document_time": get_document_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_from_sentence_time": get_token_entity_time,
            "get_token_entity_from_document_time": get_token_entity_from_document_time,
            "get_data_time": get_data_time,
            "iter_time": iter_time
        }
        print("datapack count:", packcnt)
        print("sentence count:", sentcnt)
        print("tokens count", token_sent_cnt, token_doc_cnt)
        return time_dict

    def data_processing(self):
        tokenize_time = 0
        get_document_time = 0
        get_sentence_time = 0
        get_token_entity_time = 0
        get_data_time = 0
        iter_time = 0
        get_token_entity_from_document_time = 0
        # get processed pack from dataset
        bg = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        # print("length of iterator: ", len(iter))
        process_time = time.time() - bg

        """
        iter_time:
        Iterate through the dataset and get each datapack
        """
        it = time.time()
        packcnt = 0
        sentcnt = 0
        token_sent_cnt = 0
        token_doc_cnt = 0
        for pack in iter:
            packcnt += 1
            iter_time += time.time() - it
            
            """
                get_document_time:
                Get document from databack (expected to have only one document each pack)
            """
            dt = time.time()
            document = []
            for doc in pack.get(Document):
                document.append(doc)
            if len(document) > 1:
                raise RuntimeError("More than one document in a datatuple")
            document = document[0]
            document_text = document.text
            get_document_time += time.time() - dt

            """
                get_sentence_time:
                Get sentence from databack
            """
            st = time.time()
            sentences = []
            for sent in pack.get(Sentence):
                sentcnt += 1
                sent_text = sent.text
                sentences.append(sent_text)
                get_sentence_time += time.time()-st
                
                tet = time.time()
                tokenized_tokens = []
                tokens = [t.text for t in pack.get(Token, sent)]
                entities = [entity.text
                    for entity in pack.get(EntityMention, sent)
                ]
                get_token_entity_time += time.time() - tet

                """
                tokenize_time: 
                Tokenize tokens and entities with BERTTokenizer
                """
                tt = time.time()
                tokenized_tokens += self.tokenizer.map_text_to_id(sent_text)
                token_ids = self.tokenizer.map_token_to_id(tokens)
                entity_ids = self.tokenizer.map_token_to_id(entities)
                tokenize_time += time.time()-tt

                st = time.time()
            
            """
            get_token_entity_from_document_time:
            Get token and entity from the whole document
            """
            tokdocTime = time.time()
            tokenized_tokens = []
            tokens = [t.text for t in pack.get(Token, document)]
            token_doc_cnt += len(tokens)
            entities = [entity.text
                for entity in pack.get(EntityMention, document)
            ]
            get_token_entity_from_document_time += time.time() - tokdocTime

            
            """
            get_data_time:
            Get sentence attributes with get_data request
            """
            datat = time.time()
            sentences = []
            request = {
                Sentence: ["speaker", "part_id"]
            }
            for sent in pack.get_data(Sentence, request):
                sentences.append(sent['context'])
            
            sentences_text_comp = " ".join(sentences)
            # assert(sentences_text == sentences_text_comp)
            get_data_time += time.time() - datat
            it = time.time()
        
        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_document_time": get_document_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_from_sentence_time": get_token_entity_time,
            "get_token_entity_from_document_time": get_token_entity_from_document_time,
            "get_data_time": get_data_time,
            "iter_time": iter_time
        }
        print("datapack count:", packcnt)
        print("sentence count:", sentcnt)
        print("tokens count", token_sent_cnt, token_doc_cnt)
        return time_dict

if __name__ == "__main__":
    data_pipeline = OntonoteDataProcessing()
    t1 = time.time()
    time_dict = data_pipeline.whole_data_processing()
    t2 = time.time()
    print("total time spent: ", t2-t1)
    print(f"profiling time: {time_dict}")
