from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
from os import path as osp

from tqdm import tqdm
import bs4
from bs4 import BeautifulSoup as bs
from transformers import BasicTokenizer

# NOTE: Taken directly from author code so we can use preprocessed data.
class InputFeatures(object):
  r"""
  The Container for the Features of Input Doc Spans.
  Arguments:
      unique_id (int): the unique id of the input doc span.
      example_index (int): the index of the corresponding SRC Example of the input doc span.
      page_id (str): the id of the corresponding web page of the question.
      doc_span_index (int): the index of the doc span among all the doc spans which corresponding to the same SRC
                            Example.
      tokens (list[str]): the sub-tokens of the input sequence, including cls token, sep tokens, and the sub-tokens
                          of the question and HTML file.
      token_to_orig_map (dict[int, int]): the mapping from the HTML file's sub-tokens in the sequence tokens (tokens)
                                          to the origin tokens (all_tokens in the corresponding SRC Example).
      token_is_max_context (dict[int, bool]): whether the current doc span contains the max pre- and post-context for
                                              each HTML file's sub-tokens.
      input_ids (list[int]): the ids of the sub-tokens in the input sequence (tokens).
      input_mask (list[int]): use 0/1 to distinguish the input sequence from paddings.
      segment_ids (list[int]): use 0/1 to distinguish the question and the HTML files.
      paragraph_len (int): the length of the HTML file's sub-tokens.
      start_position (int): the position where the answer starts in the input sequence (0 if the answer is not fully
                            in the input sequence).
      end_position (int): the position where the answer ends in the input sequence; NOTE that the answer tokens
                          include the token at end_position (0 if the answer is not fully in the input sequence).
      token_to_tag_index (list[int]): the mapping from sub-tokens of the input sequence to the id of the deepest tag
                                      it belongs to.
      is_impossible (bool): whether the answer is fully in the doc span.
  """

  def __init__(self,
               unique_id,
               example_index,
               page_id,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               paragraph_len,
               start_position=None,
               end_position=None,
               token_to_tag_index=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.page_id = page_id
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.start_position = start_position
    self.end_position = end_position
    self.token_to_tag_index = token_to_tag_index
    self.is_impossible = is_impossible