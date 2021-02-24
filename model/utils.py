#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import heapq
import random
import pathlib
import torch
import numpy as np
abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config


def timer(module):
    """统计函数耗时的装饰器
    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper


def simple_tokenizer(text):
    return text.split()


def count_words(counter, text):
    '''统计词频'''
    for sentence in text:
        for word in sentence:
            counter[word] += 1


def sort_batch_by_len(data_batch):
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [_tensor[i] for i in sorted_indices]
        for name, _tensor in res.items()
    }
    return data_batch


def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words, including mapping in-source OOVs from
        their temporary ids to the original OOV string (applicable in
        pointer-generator mode).
        Args:
            id_list: list of ids (integers)
            vocab: Vocabulary object
            source_oovs:
                list of OOV words (strings) in the order corresponding to
                their temporary source OOV ids (that have been assigned in
                pointer-generator mode), or None (in baseline mode)
        Returns:
            words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]  # might be [UNK]
        except IndexError:           # w is OOV
            assert_msg = "Error: cannot find the ID the in the vocabulary."
            assert source_oovs is not None, assert_msg
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        words.append(w)
    return ' '.join(words)


def source2ids(source_words, vocab):
    """映射原文tocken到对应的idx, 并且返回原文oov列表.
    Args:
        source_words: list of words (strings)
        vocab: Vocabulary object
    Returns:
        ids:
        A list of word ids (integers); OOVs are represented by their temporary
        source OOV number. If the vocabulary size is 50k and the source has 3
        OOVs tokens, then these temporary OOV numbers will be 50000, 50001,
        50002.
    oovs:
        A list of the OOV words in the source (strings), in the order
        corresponding to their temporary source OOV numbers.
    """
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        # 如果w是oov 添加到oovs 词表
        if i == unk_id:       
            if w not in oovs:  
                oovs.append(w)
            oov_num = oovs.index(w)
            # 如果50000是第一个OOV的idx, 50001则是第二个OOV的idx
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, source_oovs):
    """映射参考摘要中的token至idx，原文中的出现的OOV tokens 将被保留.
    Args:
        abstract_words (list): Tokens in the reference.
        vocab (vocab.Vocab): The vocabulary.
        source_oovs (list): OOV tokens in the source.

    Returns:
        list: The reference with tokens mapped into ids.
    """
    ids = []
    unk_id = vocab.UNK
    for w in abstract_words:
        i = vocab[w]
        # 如果是oov词汇
        if i == unk_id: 
            # 如果是原文oov
            if w in source_oovs:  
                # 找到oov对应的扩展词表中的idx
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            # 如果不是原文oov，用unk idx
            else: 
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        The scores are calculated according to the definitions in
        https://opennmt.net/OpenNMT/translation/beam_search/.
        1. Lenth normalization is used to normalize the cumulative score
        of a whole sequence.
        2. Coverage normalization is used to favor the sequences that fully
        cover the information in the source. (In this case, it serves different
        purpose from the coverage mechanism defined in PGN.)
        3. Alpha and beta are hyperparameters that used to control the
        strengths of ln and cn.
        """
        len_Y = len(self.tokens)
        # Lenth normalization
        ln = (5+len_Y)**config.alpha / (5+1)**config.alpha
        cn = config.beta * torch.sum(  # Coverage normalization
            torch.log(
                config.eps +
                torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.DEVICE))
                )
            )
        )

        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.

    Args:
        heap (list): The list to heapify.
        item (tuple):
            The tuple as item to store.
            Comparsion will be made according to values in the first position.
            If there is a tie, values in the second position will be compared,
            and so on.
        k (int): The capacity of the heap.
    """
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


def replace_oovs(in_tensor, vocab):
    """用<UNK> token的idx 替换oov idx. [2, 4, 1, unk_id, 6, unk_id]

    Args:
        in_tensor (Tensor): The tensor before replacement.
        vocab (vocab.Vocab): The vocabulary.

    Returns:
        Tensor: The tensor after replacement.
    """    
    oov_token = torch.full(in_tensor.shape, vocab.UNK).long().to(config.DEVICE)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor


class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        """根据一定的概率选择是否执行teacher_force
        Args:
            phase (int): probability level  if phase = 0, 100% teacher_forcing ,
                         phase = self.phases - 1, 0% teacher_forcing 
        Returns:
            bool: teacher_forcing or not 
        """
        sampling_prob = random.random()   # [0, 1)
        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False


def config_info(config):
    """打印配置信息
    Args:
        config (model): define in model/config.py
    Returns:
        string: config information
    """
    info = 'model_name={}, pointer={}, coverage={}, fine_tune={}, scheduled_sampling={}, weight_tying={}, ' +\
          'source={} '
    return (info.format(config.model_name, config.pointer, config.coverage, config.fine_tune, config.scheduled_sampling,
                      config.weight_tying, config.source))

