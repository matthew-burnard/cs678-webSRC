import pandas as pd
import re
import json

import os

from tqdm import tqdm

data_filepath="./data/"

def get_html_from_file(filepath):
  with open(filepath, 'r') as file:
    output = []
    for line in file.readlines():
      #print(line + "\n")
      line = re.sub('[^0-9a-zA-Z<>\' /\"]+', '', line)
      token = []
      in_tag=False
      ignoring_chars=False
      # lets scan through the html line by line
      # the idea here is we want to
      for char in line:
        if in_tag: 
          if ignoring_chars:
            if char=='>':
              in_tag=False
              ignoring_chars=False
              token.append(char)
              output.append("".join(token))
              token=[]
            else:
              continue
          else:
            if char==' ':
              ignoring_chars=True
            elif char=='>':
              in_tag=False
              token.append(char)
              output.append("".join(token))
              token=[]
            else:
              token.append(char)
        else:
          if char=='<':
            token.append(char)
            in_tag=True
          elif token=='/':
            continue #skipping these outside of tags
          elif char==' ':
            if token != []:
              output.append("".join(token))
              token=[]
            else:
              continue
          else:
            token.append(char)
  #return output
  return " ".join(output)

def test_get_html_from_file():
  filepath = "./data/auto/01/processed_data/0100001.html"
  print(get_html_from_file(filepath))

#test_get_html_from_file()

#print(train_json['data'][0]['websites'][0]['qas'][0]['answers'])

def build_dataframe(filepath, verbose=True):
  json_obj = json.load(open(train_filepath))
  cols = ['text', 'question', 'answer']
  rows = []
  total = 0
  for domain in json_obj['data']:
    for website in domain['websites']:
      total += len(website['qas'])
  
  if verbose:
    print(f"Building dataframe from {filepath}")
    pbar=tqdm(total=total)
  for domain in json_obj['data']:
    domain_name=domain['domain']
    # Get the list of answers paired with question ids
    qa_frame=[]
    for (root,dirs,files) in os.walk(data_filepath+domain_name):
      for dir in dirs:
        csv_filepath=root+"/"+dir+"/dataset.csv"
        qa_frame.append(pd.read_csv(csv_filepath))
      break
    qa_frame=pd.concat(qa_frame)
    for website in domain['websites']:
      page_id=website['page_id']
      page_filepath = data_filepath+domain_name+"/" + page_id[:2] + "/processed_data/" + page_id + ".html"
      text=get_html_from_file(page_filepath)
      for qa in website['qas']:
        id=qa['id']
        answer=qa_frame.loc[qa_frame['id']==qa['id']]['answer'].values[0]
        question = qa['question']
        rows.append([text,question,answer])
        if verbose:
          pbar.update(1)
    if verbose:
      pbar.close()
    dataframe=pd.DataFrame(data=rows,columns=cols)
    return dataframe

def save_dataframe(dataframe, filepath):
  dataframe.to_csv(filepath)

def load_dataframe(filepath):
  return pd.read_csv(filepath)

if __name__=="__main__":
  train_filepath = "./data/websrc1.0_train_.json"
  dev_filepath = "./data/websrc1.0_dev_.json"
  test_filepath = "./data/websrc1.0_test_.json"
  
  train_dataframe_path="./data.train"
  dev_dataframe_path="./data.dev"
  
  train_data = build_dataframe(train_filepath)
  #save_dataframe(train_data, train_dataframe_path)
  dev_data = build_dataframe(dev_filepath)
  #save_dataframe(dev_data, dev_dataframe_path)