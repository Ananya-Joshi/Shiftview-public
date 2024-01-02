# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

#startup ass sessinos
import streamlit as st
import numpy as np
import os, urllib
import torch 
from google.cloud import ndb
from google.cloud.ndb import model as model2
import json
import requests
from os import path
import shutil
from google.cloud import datastore
import time
import datetime
import threading
import os.path
from streamlit.ReportThread import add_report_ctx
import torch
client = ndb.Client()
curr_dir = os.getcwd()


from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}


def spin_cursor():
    q = threading.currentThread()
    my_placeholder2 = st.empty()
    my_bar2 = st.progress(0)
    i = 0 
    while getattr(q, "do_run2", True):
        if i <= 250:
            my_placeholder2.text(str(250-i) + " seconds left")
            my_bar.progress(i*0.004)
            time.sleep(1)
    my_placeholder.text("Done")
    my_bar2.progress(1)
        # my_bar3.progress(i*0.004)

def spin_cursor2(path2, r):
    t = threading.currentThread()
    if "meta" in path2 :
        st.write("Download 2/2")
        my_bar3 = st.progress(0)
    else:
        st.write("Download 1/2")
        my_bar = st.progress(0)
    while getattr(t, "do_run", True):
        if os.path.exists(path2):
           # print(os.stat(path2).st_size)
           # print(r.headers['Content-length'])
            if "meta" in path2 :
                my_bar3.progress(os.stat(path2).st_size/int(r.headers['Content-length']))
            else:
                my_bar.progress(os.stat(path2).st_size/int(r.headers['Content-length']))
    if "meta" in path2 :
        my_bar3.progress(0.99)
    else: 
        my_bar.progress(0.99)

class MyModel(ndb.Model):
    input_txt = ndb.StringProperty()
    output_txt = ndb.TextProperty()
    mod_sel = ndb.StringProperty()
    version = ndb.IntegerProperty()
    timestmp = ndb.DateTimeProperty(auto_now_add=True)

class MyModel2(ndb.Model):
    name = ndb.StringProperty()
    query = ndb.TextProperty()
    email = ndb.StringProperty()
    timestmp = ndb.DateTimeProperty(auto_now_add=True)


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():

    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions2.md"))
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Intake Form"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Intake Form":
        readme_text.empty()
        intake()
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()


def download_file(url, path2):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
       # spin_thread2 = threading.Thread(target=spin_cursor2, args=(path2, r))
        # add_report_ctx(spin_thread2)
        # spin_thread2.start()
        with open(path2, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    # spin_thread2.do_run = False
    # spin_thread2.join()
    return local_filename


def intake():      
    name2 = st.text_input("Name", value='', key=None)
    name = name2
    email2 = st.text_input("Email", value='', key=None)
    email = email2
    query2 = st.text_input("Query", value='', key=None)
    query = query2
    blank = st.text("For immediate response, please email ananya.ashish.joshi@gmail.com.")
    if st.button('Submit'):
        if ((name != "") and (email != "")) and (query != ""):
            with client.context():
                    mod2 =  MyModel2(parent=ndb.Key("Intake", int(time.time())), name=name, query=query, email=email) 
                   # print("3")
                    mod2.put()
                    st.write("Submitted...")
# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    download_file("https://ideologicaltransformer.s3.us-east-2.amazonaws.com/pytorch_model.bin", "/app/model_output/pytorch_model.bin")
    readme_text = st.markdown(get_file_content_as_string("instructions3.md"))
    starting = st.text_input("Input your starting text:", value='', key=None)
    input_text = starting.replace(",", "").replace("-", "").replace("!", "").replace("?", "")
    # cur_mod = name_convert[model_select]
    if st.button('Submit'):
        if(input_text != ""):
            # with st.spinner('Check back in about 5 minutes and ensure you have a stable internet connection.'):
            #     # data = data_download(cur_mod)
            with st.spinner('Almost ready...'):
                # Initialize the model and tokenizer
                try:
                    model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
                except KeyError:
                    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

                tokenizer = tokenizer_class.from_pretrained("model_output")
                model = model_class.from_pretrained("model_output")
                model.to("cpu")

                 # logger.info(args)
                encoded_prompt = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
                encoded_prompt = encoded_prompt.to("cpu")

                if encoded_prompt.size()[-1] == 0:
                    input_ids = None
                else:
                    input_ids = encoded_prompt

                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=100 + len(encoded_prompt[0]),
                    temperature=0.5,
                    top_k=0,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    num_return_sequences=3,
                )

                st.success("Generated ...")
                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    generated_sequence = generated_sequence.tolist()

                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                    # Remove all text after the stop token
                    # text = text[: text.find(args.stop_token) if args.stop_token else None]

                    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                    total_sequence = (
                        input_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                    )

                    with client.context():
                        mod =  MyModel(parent=ndb.Key("Simple_Translate", int(time.time())), version=generated_sequence_idx, input_txt = input_text, output_txt=total_sequence) 
                       # print("3")
                        mod.put()

                    generated_sequences.append(total_sequence)
                    st.write(total_sequence)


                
            st.success("Done")
            # spin_thread.do_run2 = False
            # spin_thread.join()

                
            #send all the data to a database


import tempfile #1



def data_download(model_name):
    mod_dict = {"345extRep":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-251.data-00000-of-00001",
    "345extDem":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-3.data-00000-of-00001",
    "345modRep":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model.ckpt.data-00000-of-00001",
    "345modDem":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-18.data-00000-of-00001",}
    mod_dict2 = {"345extRep":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-251.meta",
    "345extDem":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-3.meta",
    "345modRep":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model.ckpt.meta",
    "345modDem":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-18.meta",}
    mod_dict3 = {"345extRep":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-251.index",
    "345extDem":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-3.index",
    "345modRep":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model.ckpt.index",
    "345modDem":"https://ideologicaltransformer.s3.us-east-2.amazonaws.com/model-18.index",}
    url_file = mod_dict[model_name]
    path2 =  "/app/model-1.data-00000-of-00001"
    url_file2 = mod_dict2[model_name]
    path3 = "/app/model-1.meta"
    url_file3 = mod_dict3[model_name]
    path4 = "/app/model-1.index"

    print(path3)

    download_file(url_file, path2)
    # download_file(url_file2, path3)
    download_file(url_file3, path4)
   # print(url_file.split("/")[-1].split(".")[0])
    return url_file.split("/")[-1].split(".")[0]







# # Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    response = open(path, "r")
    return response.read()



if __name__ == "__main__":
    datastore_client = datastore.Client()

    # if not firebase_admin._apps:
    #     cred = credentials.Certificate('/Users/ananyaaccount/Downloads/ShiftView-bba67cc3dc1c.json') 
    #     default_app = firebase_admin.initialize_app(cred)
    # const serviceAccount = require('../../');
    # firebase_admin.initializeApp({
    #   credential: admin.credential.cert(serviceAccount)
    # });

    # admin.initializeApp();
    main()
