import constants
from collections import Counter, deque, defaultdict, OrderedDict
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import datetime
import numpy as np
import abc
import heapq


class CacheAlgorithm(abc.ABC):
    @abc.abstractmethod
    def run_algorithm(self, blocktrace):
        pass

    @abc.abstractmethod
    def header(self):
        pass


'''
OPT algorithm
'''


class OPT(CacheAlgorithm):
    def __init__(self, cache_size=10):
        self.cache_size = cache_size

    def run_algorithm(self, blocktrace: list):
        cache = set()
        size = len(blocktrace)
        '''#recency tracker
        recency=deque()
        #frequency tracker
        cache_frequency = defaultdict(int)
        frequency = defaultdict(int)
        '''
        # opt datastrcutre
        maxpos = size * 3
        opt_tracker = defaultdict(deque)
        position_tracker = defaultdict(int)

        for i, block in enumerate(tqdm(blocktrace, desc="Opt tracker: building index", disable=True)):
            opt_tracker[block].append(i)

        # distinct_size=set()
        # for block in tqdm(blocktrace):
        #    distinct_size.add(block)
        # print(len(distinct_size)) #2057532

        # training set:
        # prev_strategy=constants.LRU_LABEL
        # output_file=open(constants.OPT_OUTPUT_FILENAME+"_cache_size_"+str(constants.CACHE_SIZE),"w")

        hit, miss = 0, 0
        threshold = 0.0
        # start_time=datetime.datetime.now()
        # print(str(datetime.datetime.now())+":Start Loop")
        for i, block in enumerate(tqdm(blocktrace, disable=True)):
            # if i %100000==0:
            #    print(str(i)+"/"+str(size)+":"+str(round(i/size,2))+" time:"+str(datetime.datetime.now()-start_time))

            if len(opt_tracker[block]) is not 0 and opt_tracker[block][0] == i:
                opt_tracker[block].popleft()

            '''frequency[block] += 1'''
            if block in cache:
                '''cache_frequency[block] += 1
                recency.remove(block)
                recency.append(block)'''

                hit += 1
                '''if len(cache)==self.cache_size:
                    threshold+=1
                else:
                    threshold=0.0'''

                del position_tracker[i]
                if len(opt_tracker[block]) is not 0:
                    position_tracker[opt_tracker[block][0]] = block
                    opt_tracker[block].popleft()
                else:
                    position_tracker[maxpos] = block
                    maxpos -= 1

            elif len(cache) < self.cache_size:
                cache.add(block)

                '''cache_frequency[block] += 1

                recency.append(block)'''

                miss += 1

                if len(opt_tracker[block]) is not 0:
                    position_tracker[opt_tracker[block][0]] = block
                    opt_tracker[block].popleft()
                else:
                    position_tracker[maxpos] = block
                    maxpos -= 1
            else:
                evictpos = max(position_tracker)
                opt_victim = position_tracker[evictpos]

                '''#get lru victim
                lru_victim=recency[0]
                

                #get lfu victim
                lfu_victim, f = min(cache_frequency.items(), key=lambda a: a[1])
        
                value=threshold/(threshold+1.0) '''
                '''if opt_victim==lru_victim:
                    output_file.write(str(constants.LRU_LABEL)+","+str(value)+"\n") 
                elif opt_victim==lfu_victim:
                    output_file.write(str(constants.LFU_LABEL)+","+str(value)+"\n")      
                elif value>0:
                    output_file.write(str(constants.LRU_LABEL)+","+str(value)+"\n")'''
                cache.remove(opt_victim)
                cache.add(block)
                '''
                recency.remove(opt_victim)
                recency.append(block)
                cache_frequency.pop(opt_victim)
                cache_frequency[block] = frequency[block]'''
                threshold = 0.0
                miss += 1

                del position_tracker[evictpos]
                if len(opt_tracker[block]) is not 0:
                    position_tracker[opt_tracker[block][0]] = block
                    opt_tracker[block].popleft()
                else:
                    position_tracker[maxpos] = block
                    maxpos -= 1
        # output_file.close()
        return hit / (hit + miss)

    def header(self):
        return (type(self).__name__)


'''
This is not used, just keep for records
'''


def opt_2(blocktrace: list, cache_size=constants.CACHE_SIZE):
    cache = set()
    size = len(blocktrace)
    # recency tracker
    recency = deque()
    # frequency tracker
    cache_frequency = defaultdict(int)
    frequency = defaultdict(int)

    # opt datastrcutre
    maxpos = size * 3
    opt_tracker = defaultdict(deque)
    position_tracker = defaultdict(int)

    for i, block in enumerate(tqdm(blocktrace, desc="Opt tracker: building index")):
        opt_tracker[block].append(i)

    # distinct_size=set()
    # for block in tqdm(blocktrace):
    #    distinct_size.add(block)
    # print(len(distinct_size)) #2057532

    # training set:
    # prev_strategy=constants.LRU_LABEL
    output_file = open(constants.OPT_OUTPUT_FILENAME + "_cache_size_" + str(constants.CACHE_SIZE), "w")

    hit, miss = 0, 0
    threshold = 0.0
    prev_strategy = ''
    start_time = datetime.datetime.now()
    # print(str(datetime.datetime.now())+":Start Loop")
    for i, block in enumerate(tqdm(blocktrace, disable=True)):
        if i % 100000 == 0:
            print(str(i) + "/" + str(size) + ":" + str(round(i / size, 2)) + " time:" + str(
                datetime.datetime.now() - start_time))

        if len(opt_tracker[block]) is not 0 and opt_tracker[block][0] == i:
            opt_tracker[block].popleft()

        frequency[block] += 1
        if block in cache:
            cache_frequency[block] += 1
            recency.remove(block)
            recency.append(block)

            hit += 1
            if len(cache) == cache_size:
                threshold += 1
            else:
                threshold = 0.0

            del position_tracker[i]
            if len(opt_tracker[block]) is not 0:
                position_tracker[opt_tracker[block][0]] = block
                opt_tracker[block].popleft()
            else:
                position_tracker[maxpos] = block
                maxpos -= 1

        elif len(cache) < cache_size:
            cache.add(block)

            cache_frequency[block] += 1

            recency.append(block)

            miss += 1

            if len(opt_tracker[block]) is not 0:
                position_tracker[opt_tracker[block][0]] = block
                opt_tracker[block].popleft()
            else:
                position_tracker[maxpos] = block
                maxpos -= 1
        else:
            evictpos = max(position_tracker)
            opt_victim = position_tracker[evictpos]

            # get lru victim
            lru_victim = recency[0]

            # get lfu victim
            lfu_victim, f = min(cache_frequency.items(), key=lambda a: a[1])

            value = threshold / (threshold + 1.0)
            victim = lru_victim
            current_strategy = ""
            if opt_victim == lfu_victim and value > 0:
                current_strategy = str(constants.LFU_LABEL) + "," + str(value)
                victim = lfu_victim
            elif value > 0:
                current_strategy = str(constants.LRU_LABEL) + "," + str(value)

            if value > 0 and current_strategy != prev_strategy:
                output_file.write(current_strategy + "\n")
                prev_strategy = current_strategy
                threshold = 0.0

            cache.remove(victim)
            cache.add(block)
            recency.remove(victim)
            recency.append(block)
            cache_frequency.pop(victim)
            cache_frequency[block] = frequency[block]

            miss += 1

            for k, v in position_tracker.items():
                if v == victim:
                    pos = k
            del position_tracker[pos]
            if len(opt_tracker[block]) is not 0:
                position_tracker[opt_tracker[block][0]] = block
                opt_tracker[block].popleft()
            else:
                position_tracker[maxpos] = block
                maxpos -= 1
    output_file.close()
    return hit / (hit + miss)


class LRU((CacheAlgorithm)):
    def __init__(self, cache_size=10):
        self.cache_size = cache_size

    def run_algorithm(self, blocktrace):

        cache = set()
        recency = deque()
        hit, miss = 0, 0

        for block in tqdm(blocktrace, disable=True):

            if block in cache:
                recency.remove(block)
                recency.append(block)
                hit += 1

            elif len(cache) < self.cache_size:
                cache.add(block)
                recency.append(block)
                miss += 1

            else:
                cache.remove(recency[0])
                recency.popleft()
                cache.add(block)
                recency.append(block)
                miss += 1

        hitrate = hit / (hit + miss)
        return hitrate

    def header(self):
        return (type(self).__name__)


class CLOCK((CacheAlgorithm)):
    def __init__(self, cache_size=10, clock_value=1):
        self.cache_size = cache_size
        self.clock_value = clock_value

    def run_algorithm(self, blocktrace):

        cache = set()
        recency = OrderedDict()
        hit, miss = 0, 0
        for block in tqdm(blocktrace, disable=True):

            if block in cache:
                # recency.remove(block)
                # recency.append(block)
                recency[block] = self.clock_value
                recency.move_to_end(block, last=True)  # move to the right end
                hit += 1

            elif len(cache) < self.cache_size:
                cache.add(block)
                # recency.append(block)
                recency[block] = self.clock_value
                miss += 1

            else:
                evict = recency.popitem(last=False)
                while (evict[1] > 0):
                    recency[evict[0]] = evict[1] - 1
                    evict = recency.popitem(last=False)

                cache.remove(evict[0])
                cache.add(block)
                recency[block] = self.clock_value
                miss += 1

        hitrate = hit / (hit + miss)
        return hitrate

    def header(self):
        return (type(self).__name__)


class CLOCK_LFU((CacheAlgorithm)):
    def __init__(self, cache_size=10):
        self.cache_size = cache_size

    def run_algorithm(self, blocktrace):

        cache = set()
        recency = OrderedDict()
        frequency = defaultdict(int)
        hit, miss = 0, 0

        for block in tqdm(blocktrace, disable=True):
            frequency[block] += 1

            if block in cache:
                recency[block] = frequency[block]
                recency.move_to_end(block, last=True)  # move to the right end
                hit += 1

            elif len(cache) < self.cache_size:
                cache.add(block)
                recency[block] = frequency[block]
                miss += 1

            else:
                min_value = min(recency.items(), key=lambda x: x[1])[1]
                for key, val in recency.items():
                    recency[key] = val - min_value
                evict = recency.popitem(last=False)
                while (evict[1] > 0):
                    recency[evict[0]] = evict[1] - 1
                    evict = recency.popitem(last=False)

                cache.remove(evict[0])
                cache.add(block)
                recency[block] = frequency[block]
                miss += 1

        hitrate = hit / (hit + miss)
        return hitrate

    def header(self):
        return (type(self).__name__)


class MRU((CacheAlgorithm)):
    def __init__(self, cache_size=10):
        self.cache_size = cache_size

    def run_algorithm(self, blocktrace):

        cache = set()
        recency = deque()
        hit, miss = 0, 0

        for block in tqdm(blocktrace, disable=True):

            if block in cache:
                recency.remove(block)
                recency.append(block)
                hit += 1

            elif len(cache) < self.cache_size:
                cache.add(block)
                recency.append(block)
                miss += 1

            else:
                cache.remove(recency.pop())
                cache.add(block)
                recency.append(block)
                miss += 1

        hitrate = hit / (hit + miss)
        return hitrate

    def header(self):
        return (type(self).__name__)


class LFU(CacheAlgorithm):
    def __init__(self, cache_size=10):
        self.cache_size = cache_size

    def run_algorithm(self, blocktrace):
        cache = set()
        cache_frequency = defaultdict(int)
        frequency = defaultdict(int)

        hit, miss = 0, 0

        for block in tqdm(blocktrace, disable=True):
            frequency[block] += 1

            if block in cache:
                hit += 1
                cache_frequency[block] += 1

            elif len(cache) < self.cache_size:
                cache.add(block)
                cache_frequency[block] += 1
                miss += 1

            else:
                e, f = min(cache_frequency.items(), key=lambda a: a[1])
                cache_frequency.pop(e)
                cache.remove(e)
                cache.add(block)
                cache_frequency[block] = frequency[block]
                miss += 1

        hitrate = hit / (hit + miss)
        return hitrate

    def header(self):
        return (type(self).__name__)


'''
LeCar Algorithm
'''


class LeCarLruLuf:

    def __init__(self, learning_rate=0.45, discount_rate=0.005, cache_size=100):
        self.learning_rate = learning_rate
        # self.discount_rate=discount_rate ** (1/cache_size)
        self.discount_rate = discount_rate
        self.cache_size = cache_size
        self.weight_lru = 0.5
        self.weight_lfu = 0.5

        self.history_size = self.cache_size * 1
        self.cache = set()
        self.page_timestamp = dict()
        self.history_lru = deque()
        self.history_lfu = deque()
        self.global_frequency = defaultdict(int)

        self.recency = deque()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)

        self.page_time = dict()

        print("LeCarLruLfu: learning_rate:" + str(self.learning_rate) + " discount_rate:" + str(
            discount_rate) + "," + str(self.discount_rate) + " cache_size:" + str(self.cache_size))

    def run_algorithm(self, blocktrace: list, timestamp: list):
        size = len(blocktrace)

        hit, miss = 0, 0
        lru_miss = 0
        lfu_miss = 0
        total_lru_time = 0
        total_lfu_time = 0
        last_miss = 0
        for i, block in enumerate(tqdm(blocktrace, disable=True)):
            '''if i %1000==0:
                #    print(str(i)+"/"+str(size)+":"+str(round(i/size,2))+" time:"+str(datetime.datetime.now()-start_time))
                    print("reward:"+str(reward_lru)+","+str(reward_lfu))
                    print("weight:"+str(self.weight_lru)+","+str(self.weight_lfu))
             '''
            self.frequency[block] += 1
            if block in self.cache:
                self.update_block(block)
                hit += 1
            else:
                '''
                if block in self.page_timestamp:
                    time_spend_in_millis=self.millis(timestamp[i])-self.page_timestamp[block]
                else:
                    time_spend_in_millis=1'''
                if block in self.history_lru or block in self.history_lfu:
                    time_spend = i - self.page_time[block]

                    if time_spend > miss - last_miss:
                        if block in self.history_lru:
                            self.history_lru.remove(block)
                        if block in self.history_lfu:
                            self.history_lfu.remove(block)
                        last_miss = miss

                # if self.reward_lru != 1 and self.reward_lfu !=1:
                #    print("before reward:"+str(self.reward_lru)+","+str(self.reward_lfu))
                reward_lru = 0
                reward_lfu = 0

                if block in self.history_lru:
                    self.history_lru.remove(block)
                    reward_lfu = self.discount_rate ** time_spend
                    '''self.weight_lfu=self.weight_lfu* np.exp(self.learning_rate*reward_lfu)
                    self.weight_lru=self.weight_lru/(self.weight_lfu+self.weight_lru)
                    self.weight_lfu=1-self.weight_lru'''
                    self.update_weight(reward_lru, reward_lfu)
                    lru_miss += 1
                    total_lru_time += time_spend

                if block in self.history_lfu:
                    self.history_lfu.remove(block)
                    reward_lru = self.discount_rate ** time_spend
                    '''self.weight_lru=self.weight_lru* np.exp(self.learning_rate*reward_lru)
                    self.weight_lru=self.weight_lru/(self.weight_lfu+self.weight_lru)
                    self.weight_lfu=1-self.weight_lru'''
                    self.update_weight(reward_lru, reward_lfu)
                    lfu_miss += 1
                    total_lfu_time += time_spend

                if len(self.cache) == self.cache_size:
                    victim = self.evcitPage(block)
                    self.cache.remove(victim)
                    self.remove_block(victim)
                self.cache.add(block)
                self.add_block(block)
                # self.page_timestamp[block]=self.millis(timestamp[i])
                miss += 1
            self.page_time[block] = i

        print("weight:" + str(self.weight_lru) + "," + str(self.weight_lfu))
        print("lru_miss:" + str(lru_miss) + " lfu_miss:" + str(lfu_miss) + " total miss:" + str(miss))
        print("total_time_lru:" + str(total_lru_time) + " total_time_lfu:" + str(total_lfu_time))
        print("avg lru time:" + str(total_lru_time / lru_miss) + " avg lfu time:" + str(total_lfu_time / lfu_miss))

        return hit / (hit + miss)

    def update_weight(self, reward_lru, reward_lfu):
        # if self.reward_lru != 1 or self.reward_lfu !=1:

        # print("before reward:"+str(self.reward_lru)+","+str(self.reward_lfu))
        self.weight_lfu = self.weight_lfu * np.exp(self.learning_rate * reward_lfu)
        self.weight_lru = self.weight_lru * np.exp(self.learning_rate * reward_lru)
        self.weight_lru = self.weight_lru / (self.weight_lfu + self.weight_lru)
        self.weight_lfu = 1 - self.weight_lru
        # if self.reward_lru != 1 or self.reward_lfu !=1:
        #    print("after:"+str(self.weight_lru)+","+str(self.weight_lfu))
        # print("after reward:"+str(self.reward_lru)+","+str(self.reward_lfu))

    def millis(self, my_time):
        return int(round(my_time * 1000))

    def chooseRandom(self):
        r = np.random.rand()
        if r < self.weight_lru:
            return 0
        return 1

    def evcitPage(self, block):
        policy = self.chooseRandom()
        if policy == 0:
            victim = self.recency[0]
            if len(self.history_lru) == self.history_size:
                self.history_lru.popleft()
            self.history_lru.append(victim)
        else:
            victim, f = min(self.cache_frequency.items(), key=lambda a: a[1])
            if len(self.history_lfu) == self.history_size:
                self.history_lfu.popleft()
            self.history_lfu.append(victim)
        return victim

    def update_block(self, block):
        self.cache_frequency[block] += 1
        self.recency.remove(block)
        self.recency.append(block)

    def remove_block(self, block):
        self.recency.remove(block)
        self.cache_frequency.pop(block)

    def add_block(self, block):
        self.recency.append(block)
        self.cache_frequency[block] = self.frequency[block]
        # self.cache_frequency[block] = 1


'''
My algorithm that leverage the history_structure from leCar
'''


class LeCar_Opt(CacheAlgorithm):

    def __init__(self, cache_size=10, history_size_factor=1):

        self.cache_size = cache_size

        self.history_size = self.cache_size * history_size_factor
        self.cache = set()
        self.page_timestamp = dict()
        self.history_lru = deque()
        self.history_lfu = deque()
        self.global_frequency = defaultdict(int)

        self.recency = deque()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)

        self.page_time = dict()

        # print(self.__class__.__name__+": history_size:"+str(self.history_size)+" cache_size:"+str(self.cache_size))

    def run_algorithm(self, blocktrace: list):
        size = len(blocktrace)

        hit, miss = 0, 0
        lru_miss = 0
        lfu_miss = 0
        last_miss = 0
        last_hit = 0
        current_policy = 0  # 0 lru, 1 lfu
        time_counter = 0
        switch_conter = 0
        # switch_policy_counter=0
        for i, block in enumerate(tqdm(blocktrace, disable=True)):

            self.frequency[block] += 1
            if block in self.cache:
                self.update_block(block)
                hit += 1
                last_hit += 1
                # switch_policy_counter=0
            else:
                '''
                #time counter implementation, if it is in time counter, that miss in last cachesize,  remove block
                if block in self.history_lru or block in self.history_lfu:
                    time_spend=i-self.page_time[block]
                    
                    if time_counter>=self.cache_size :
                        if block in self.history_lru:
                            self.history_lru.remove(block)
                        if block in self.history_lfu:
                            self.history_lfu.remove(block)
                        time_counter=0'''

                if block in self.history_lru:
                    self.history_lru.remove(block)
                    lru_miss += 1

                if block in self.history_lfu:
                    self.history_lfu.remove(block)
                    lfu_miss += 1

                if len(self.cache) == self.cache_size:
                    # switch policy
                    '''if hit>0 and switch_policy_counter > (1.0/ (hit/(hit+miss))):
                        print("---------------------")
                        print((1.0/ (hit/(hit+miss))))
                        print(switch_policy_counter)
                        print((last_hit/(last_hit+last_miss)))
                        print((hit/(hit+miss)))'''

                    # if  hit >0  and switch_policy_counter > (1.0/ (hit/(hit+miss))): #and (last_hit/(last_hit+last_miss)) <= (hit/(hit+miss)):
                    if hit > 0:  # and (last_hit/(last_hit+last_miss)) <= (hit/(hit+miss)):
                        # print("---------------------")
                        # print((1.0/ (hit/(hit+miss))))
                        # print(switch_policy_counter)
                        # print((last_hit/(last_hit+last_miss)))
                        # print((hit/(hit+miss)))
                        # switch_policy_counter=0
                        if (current_policy == 0 and (
                                lru_miss > (1.0 / (hit / (hit + miss))) or lru_miss > self.cache_size)) or (
                                current_policy == 1 and (
                                lfu_miss > (1.0 / (hit / (hit + miss))) or lfu_miss > self.cache_size)):
                            '''print("---------------------")
                            print((1.0/ (hit/(hit+miss))))
                            print((last_hit/(last_hit+last_miss)))
                            print((hit/(hit+miss)))
                            print(current_policy)
                            print(lru_miss)
                            print(lfu_miss)'''

                            current_policy = 1 - current_policy  # switch from 0 to 1, or 1 to 0
                            last_miss = 0
                            last_hit = 0
                            lru_miss = 0
                            lfu_miss = 0
                            switch_conter += 1

                    # else:
                    # switch_policy_counter+=1

                    if current_policy == 0:
                        victim = self.recency[0]
                        if len(self.history_lru) == self.history_size:
                            self.history_lru.popleft()
                        self.history_lru.append(victim)
                    else:
                        victim, f = min(self.cache_frequency.items(), key=lambda a: a[1])
                        if len(self.history_lfu) == self.history_size:
                            self.history_lfu.popleft()
                        self.history_lfu.append(victim)

                    self.cache.remove(victim)
                    self.remove_block(victim)
                self.cache.add(block)
                self.add_block(block)
                miss += 1
                last_miss += 1
                time_counter += 1
            self.page_time[block] = i

        print("     " + self.header() + ", # of policy switches:" + str(switch_conter))
        return hit / (hit + miss)

    def header(self):
        return (type(self).__name__ + " history size:" + str(self.history_size))

    def update_block(self, block):
        self.cache_frequency[block] += 1
        self.recency.remove(block)
        self.recency.append(block)

    def remove_block(self, block):
        self.recency.remove(block)
        self.cache_frequency.pop(block)

    def add_block(self, block):
        self.recency.append(block)
        self.cache_frequency[block] = self.frequency[block]
        # self.cache_frequency[block] = 1


''' 
Lecar with time to evict
'''


class LeCar_Opt2(CacheAlgorithm):

    def __init__(self, cache_size=10, history_size_factor=1):

        self.cache_size = cache_size

        self.history_size = self.cache_size * history_size_factor
        self.cache = set()
        self.page_timestamp = dict()
        self.history_lru = deque()
        self.history_lfu = deque()
        self.global_frequency = defaultdict(int)

        self.recency = deque()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)

        self.page_time = dict()

        # print(self.__class__.__name__+": history_size:"+str(self.history_size)+" cache_size:"+str(self.cache_size))

    def run_algorithm(self, blocktrace: list):
        size = len(blocktrace)

        hit, miss = 0, 0
        lru_miss = 0
        lfu_miss = 0
        last_miss = 0
        last_hit = 0
        current_policy = 0  # 0 lru, 1 lfu
        time_counter = 0
        switch_conter = 0
        # switch_policy_counter=0
        switch_time_counter = 0
        for i, block in enumerate(tqdm(blocktrace, disable=True)):

            self.frequency[block] += 1
            if block in self.cache:
                self.update_block(block)
                hit += 1
                last_hit += 1
                # switch_policy_counter=0
            else:

                # time counter implementation, if it is in time counter, that miss in last cachesize,  remove block
                if block in self.history_lru or block in self.history_lfu:
                    time_spend = i - self.page_time[block]

                    if time_counter >= self.cache_size:
                        if block in self.history_lru:
                            self.history_lru.remove(block)
                        if block in self.history_lfu:
                            self.history_lfu.remove(block)
                        time_counter = 0

                if block in self.history_lru:
                    self.history_lru.remove(block)
                    lru_miss += 1

                if block in self.history_lfu:
                    self.history_lfu.remove(block)
                    lfu_miss += 1

                if len(self.cache) == self.cache_size:
                    # switch policy
                    '''if hit>0 and switch_policy_counter > (1.0/ (hit/(hit+miss))):
                        print("---------------------")
                        print((1.0/ (hit/(hit+miss))))
                        print(switch_policy_counter)
                        print((last_hit/(last_hit+last_miss)))
                        print((hit/(hit+miss)))'''

                    # if  hit >0  and switch_policy_counter > (1.0/ (hit/(hit+miss))): #and (last_hit/(last_hit+last_miss)) <= (hit/(hit+miss)):
                    if hit > 0:  # and (last_hit/(last_hit+last_miss)) <= (hit/(hit+miss)):
                        # print("---------------------")
                        # print((1.0/ (hit/(hit+miss))))
                        # print(switch_policy_counter)
                        # print((last_hit/(last_hit+last_miss)))
                        # print((hit/(hit+miss)))
                        # switch_policy_counter=0
                        '''
                        if (current_policy==0 and (lru_miss > (1.0/ (hit/(hit+miss))) or lru_miss > self.cache_size ) ) or (current_policy==1 and (lfu_miss > (1.0/ (hit/(hit+miss))) or  lfu_miss > self.cache_size ) ):
                            current_policy=1-current_policy #switch from 0 to 1, or 1 to 0
                            last_miss=0
                            last_hit=0
                            lru_miss=0
                            lfu_miss=0
                            switch_conter+=1'''
                        if switch_time_counter > self.cache_size:
                            if (current_policy == 0 and lru_miss / (last_miss + lru_miss) > (
                                    last_hit / (last_hit + last_miss))) or (
                                    current_policy == 1 and lfu_miss / (last_miss + lfu_miss) > (
                                    last_hit / (last_hit + last_miss))):
                                switch_time_counter = 0
                                current_policy = 1 - current_policy  # switch from 0 to 1, or 1 to 0
                                last_miss = 0
                                last_hit = 0
                                lru_miss = 0
                                lfu_miss = 0
                                switch_conter += 1

                    # else:
                    # switch_policy_counter+=1

                    if current_policy == 0:
                        victim = self.recency[0]
                        if len(self.history_lru) == self.history_size:
                            self.history_lru.popleft()
                        self.history_lru.append(victim)
                    else:
                        victim, f = min(self.cache_frequency.items(), key=lambda a: a[1])
                        if len(self.history_lfu) == self.history_size:
                            self.history_lfu.popleft()
                        self.history_lfu.append(victim)

                    self.cache.remove(victim)
                    self.remove_block(victim)
                self.cache.add(block)
                self.add_block(block)
                miss += 1
                last_miss += 1
                time_counter += 1
                switch_time_counter += 1
            self.page_time[block] = i

        print("     " + self.header() + ", # of policy switches:" + str(switch_conter))
        return hit / (hit + miss)

    def header(self):
        return (type(self).__name__ + " history size:" + str(self.history_size))

    def update_block(self, block):
        self.cache_frequency[block] += 1
        self.recency.remove(block)
        self.recency.append(block)

    def remove_block(self, block):
        self.recency.remove(block)
        self.cache_frequency.pop(block)

    def add_block(self, block):
        self.recency.append(block)
        self.cache_frequency[block] = self.frequency[block]
        # self.cache_frequency[block] = 1


'''
My algorithm that leverage the history_structure from leCar
'''


class LeCar_Opt3(CacheAlgorithm):

    def __init__(self, cache_size=10, history_size_factor=1):

        self.cache_size = cache_size

        self.history_size = self.cache_size * history_size_factor
        self.cache = set()
        self.page_timestamp = dict()
        # self.history_lru = deque()
        # self.history_lfu = deque()
        self.history_lru = OrderedDict()
        self.history_lfu = OrderedDict()
        self.global_frequency = defaultdict(int)

        self.recency = deque()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)

        self.page_time = dict()

        # print(self.__class__.__name__+": history_size:"+str(self.history_size)+" cache_size:"+str(self.cache_size))

    def run_algorithm(self, blocktrace: list):
        size = len(blocktrace)

        hit, miss = 0, 0
        lru_miss = 0
        lfu_miss = 0

        current_policy = 0  # 0 lru, 1 lfu
        time_counter = 0
        switch_conter = 0
        # switch_policy_counter=0
        for i, block in enumerate(tqdm(blocktrace, disable=True)):

            self.frequency[block] += 1
            if block in self.cache:
                self.update_block(block)
                hit += 1

                time_counter -= 1
            else:

                # time counter implementation, if it is in time counter, that miss in last cachesize,  remove block
                if block in self.history_lru or block in self.history_lfu:
                    # time_spend=i-self.page_time[block]

                    if time_counter >= self.cache_size:
                        if block in self.history_lru:
                            del self.history_lru[block]
                        if block in self.history_lfu:
                            del self.history_lfu[block]
                        time_counter = 0

                if block in self.history_lru and current_policy == 0:
                    self.history_lru.pop(block)
                    current_policy = 1 - current_policy
                    switch_conter += 1
                    lru_miss += 1

                if block in self.history_lfu and current_policy == 1:
                    self.history_lfu.pop(block)
                    current_policy = 1 - current_policy
                    switch_conter += 1
                    lfu_miss += 1

                if len(self.cache) == self.cache_size:
                    if current_policy == 0:
                        victim = self.recency[0]
                        if len(self.history_lru) == self.history_size:
                            self.history_lru.popitem(last=False)
                        self.history_lru[victim] = 0
                    else:
                        victim, f = min(self.cache_frequency.items(), key=lambda a: a[1])
                        if len(self.history_lfu) == self.history_size:
                            self.history_lfu.popitem(last=False)
                        self.history_lfu[victim] = 0

                    self.cache.remove(victim)
                    self.remove_block(victim)
                self.cache.add(block)
                self.add_block(block)
                miss += 1
                time_counter += 1
            # self.page_time[block] = i

        print("     " + self.header() + ", # of policy switches:" + str(switch_conter))
        return hit / (hit + miss)

    def header(self):
        return (type(self).__name__ + " history size:" + str(self.history_size))

    def update_block(self, block):
        self.cache_frequency[block] += 1
        self.recency.remove(block)
        self.recency.append(block)

    def remove_block(self, block):
        self.recency.remove(block)
        self.cache_frequency.pop(block)

    def add_block(self, block):
        self.recency.append(block)
        self.cache_frequency[block] = self.frequency[block]
        # self.cache_frequency[block] = 1


class LeCar_Opt4(CacheAlgorithm):

    def __init__(self, cache_size=10, history_size_factor=1):

        self.cache_size = cache_size

        self.history_size = self.cache_size * history_size_factor
        self.cache = set()
        self.page_timestamp = dict()
        # self.history_lru = deque()
        # self.history_lfu = deque()
        self.history_lru = OrderedDict()
        self.history_lfu = OrderedDict()
        self.global_frequency = defaultdict(int)

        #self.recency = deque()
        self.recency = OrderedDict()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)

        self.page_time = dict()

        # print(self.__class__.__name__+": history_size:"+str(self.history_size)+" cache_size:"+str(self.cache_size))

    def run_algorithm(self, blocktrace: list):
        size = len(blocktrace)

        hit, miss = 0, 0
        lru_miss = 0
        lfu_miss = 0

        current_policy = 0  # 0 lru, 1 lfu
        time_counter = 0
        switch_conter = 0
        # switch_policy_counter=0
        for i, block in enumerate(tqdm(blocktrace, disable=True)):

            self.frequency[block] += 1
            if block in self.cache:
                self.update_block(block)
                hit += 1

                #time_counter -= 1
            else:

                # time counter implementation, if it is in time counter, that miss in last cachesize,  remove block
                if block in self.history_lru and current_policy == 0:
                    max_value = max(self.cache_frequency.items(), key=lambda x: x[1])[1]
                    if self.frequency[block] < max_value:
                        del self.history_lru[block]
                        time_counter = 0
                elif block in self.history_lfu and current_policy == 1:
                    time_spend=i-self.page_time[block]
                    if time_spend >=self.cache_size:
                        del self.history_lfu[block]
                    '''if time_counter > self.cache_size:
                        del self.history_lfu[block]
                        time_counter = 0'''


                if block in self.history_lru and current_policy == 0:
                    self.history_lru.pop(block)
                    current_policy = 1 - current_policy
                    switch_conter += 1
                    lru_miss += 1

                if block in self.history_lfu and current_policy == 1:
                    self.history_lfu.pop(block)
                    current_policy = 1 - current_policy
                    switch_conter += 1
                    lfu_miss += 1
                flag = True
                if len(self.cache) == self.cache_size:
                    if current_policy == 0:
                        victim = self.recency.popitem(last=False)[0]
                        flag=False
                        if len(self.history_lru) == self.history_size:
                            self.history_lru.popitem(last=False)
                        self.history_lru[victim] = 0
                    else:
                        victim, f = min(self.cache_frequency.items(), key=lambda a: a[1])
                        if len(self.history_lfu) == self.history_size:
                            self.history_lfu.popitem(last=False)
                        self.history_lfu[victim] = 0

                    self.cache.remove(victim)
                    self.remove_block(victim,flag)
                self.cache.add(block)
                self.add_block(block)
                miss += 1
                time_counter += 1
            self.page_time[block] = i

        print("     " + self.header() + ", # of policy switches:" + str(switch_conter))
        return hit / (hit + miss)

    def header(self):
        return (type(self).__name__ + " history size:" + str(self.history_size))

    def update_block(self, block):
        self.cache_frequency[block] += 1
        #self.recency.pop(block)
        #self.recency[block]=0
        self.recency.move_to_end(block)

    def remove_block(self, block, flag):
        if flag:
            self.recency.pop(block)
        self.cache_frequency.pop(block)

    def add_block(self, block):
        self.recency[block]=0
        self.cache_frequency[block] = self.frequency[block]
        # self.cache_frequency[block] = 1


class LeCar_Opt5(CacheAlgorithm):

    def __init__(self, cache_size=10, history_size_factor=1):

        self.cache_size = cache_size

        self.history_size = self.cache_size * history_size_factor
        self.cache = set()
        self.page_timestamp = dict()
        # self.history_lru = deque()
        # self.history_lfu = deque()
        self.history_lru = OrderedDict()
        self.history_lfu = OrderedDict()
        self.global_frequency = defaultdict(int)

        #self.recency = deque()
        self.recency = OrderedDict()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)

        self.page_time = dict()

        # print(self.__class__.__name__+": history_size:"+str(self.history_size)+" cache_size:"+str(self.cache_size))

    def run_algorithm(self, blocktrace: list):
        size = len(blocktrace)

        hit, miss = 0, 0
        lru_miss = 0
        lfu_miss = 0

        current_policy = 0  # 0 lru, 1 lfu
        time_counter = 0
        switch_counter = 0
        # switch_policy_counter=0
        for i, block in enumerate(tqdm(blocktrace, disable=True)):

            self.frequency[block] += 1
            if block in self.cache:
                self.update_block(block)
                hit += 1

                #time_counter -= 1
            else:

                # time counter implementation, if it is in time counter, that miss in last cachesize,  remove block
                if block in self.history_lru and current_policy == 0:
                    max_value = max(self.cache_frequency.items(), key=lambda x: x[1])[1]
                    if self.frequency[block] < max_value:
                        del self.history_lru[block]
                        time_counter = 0
                elif block in self.history_lfu and current_policy == 1:
                    time_spend=i-self.page_time[block]
                    if time_spend>=self.cache_size:
                        del self.history_lfu[block]
                    '''if time_counter > self.cache_size:
                        del self.history_lfu[block]
                        time_counter = 0'''


                if block in self.history_lru and current_policy == 0:
                    self.history_lru.pop(block)
                    current_policy = 1 - current_policy
                    switch_counter += 1
                    lru_miss += 1

                if block in self.history_lfu and current_policy == 1:
                    # sort keys by values and replace recency
                    #self.recency = sorted(self.cache_frequency, key=self.cache_frequency.get)
                    self.recency = OrderedDict(sorted(self.cache_frequency.items(),key=lambda x: x[1]))
                    self.history_lfu.pop(block)
                    current_policy = 1 - current_policy
                    switch_counter += 1
                    lfu_miss += 1
                flag=True
                if len(self.cache) == self.cache_size:
                    if current_policy == 0:
                        victim = self.recency.popitem(last=False)[0]
                        flag=False
                        if len(self.history_lru) == self.history_size:
                            self.history_lru.popitem(last=False)
                        self.history_lru[victim] = 0
                    else:
                        victim, f = min(self.cache_frequency.items(), key=lambda a: a[1])
                        if len(self.history_lfu) == self.history_size:
                            self.history_lfu.popitem(last=False)
                        self.history_lfu[victim] = 0

                    self.cache.remove(victim)
                    self.remove_block(victim,flag)
                self.cache.add(block)
                self.add_block(block)
                miss += 1
                time_counter += 1
            self.page_time[block] = i

        print("     " + self.header() + ", # of policy switches:" + str(switch_counter))
        return hit / (hit + miss)

    def header(self):
        return (type(self).__name__ + " history size:" + str(self.history_size))

    def update_block(self, block):
        self.cache_frequency[block] += 1
        #self.recency.pop(block)
        #self.recency[block]=0
        self.recency.move_to_end(block)

    def remove_block(self, block,flag):
        if flag:
            self.recency.pop(block)
        self.cache_frequency.pop(block)

    def add_block(self, block):
        self.recency[block]=0
        self.cache_frequency[block] = self.frequency[block]
        # self.cache_frequency[block] = 1

'''
Paper:
https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf

Patented by IBM:
https://web.archive.org/web/20170729195309/http://patft1.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm&r=1&f=G&l=50&s1=6996676.PN.&OS=PN/6996676&RS=PN/6996676

reference:
https://github.com/sarantarnoi/ARCwithPython/blob/master/MyARC.py
https://gist.github.com/pior/da3b6268c40fa30c222f
'''


class ARC(CacheAlgorithm):
    def __init__(self, size):
        self.cache = set()
        self.cache_size = size
        self.recency_cache_size = 0  # lru_cache size
        self.recency_cache = deque()
        self.ghost_recency_cache = deque()
        self.frequency_cache = deque()
        self.ghost_frequency_cache = deque()
        self.hit = 0
        self.miss = 0

    def replace(self, item):

        if len(self.recency_cache) >= 1 and (
                (item in self.ghost_frequency_cache and len(self.recency_cache) == self.recency_cache_size) or len(
            self.recency_cache) > self.recency_cache_size):
            old = self.recency_cache.pop()
            self.ghost_recency_cache.appendleft(old)
        else:
            old = self.frequency_cache.pop()
            self.ghost_frequency_cache.appendleft(old)

        self.cache.remove(old)

    def re(self, item):
        # Case I
        if (item in self.recency_cache) or (item in self.frequency_cache):
            if item in self.recency_cache:
                self.recency_cache.remove(item)

            elif item in self.frequency_cache:
                self.frequency_cache.remove(item)

            self.frequency_cache.appendleft(item)
            self.hit += 1
        # Case II
        elif (item in self.ghost_recency_cache):

            self.recency_cache_size = min(self.cache_size, self.recency_cache_size + max(
                len(self.ghost_frequency_cache) / len(self.ghost_recency_cache) * 1., 1))
            self.replace(item)
            self.ghost_recency_cache.remove(item)
            self.frequency_cache.appendleft(item)
            self.cache.add(item)
            self.miss += 1

        # Case III 
        elif (item in self.ghost_frequency_cache):

            self.recency_cache_size = max(0, self.recency_cache_size - max(
                len(self.ghost_recency_cache) / len(self.ghost_frequency_cache) * 1., 1))
            self.replace(item)
            self.ghost_frequency_cache.remove(item)
            self.frequency_cache.appendleft(item)
            self.cache.add(item)
            self.miss += 1

        # Case IV (Inserting a new item)
        elif (item not in self.recency_cache) or (item not in self.ghost_recency_cache) or (
                item not in self.frequency_cache) or (item not in self.ghost_frequency_cache):

            # Case (i)
            if len(self.recency_cache) + len(self.ghost_recency_cache) == self.cache_size:
                # "in Case IV(i)"
                if len(self.recency_cache) < self.cache_size:
                    self.ghost_recency_cache.pop()
                    self.replace(item)
                else:
                    old = self.recency_cache.pop()
                    self.cache.remove(old)

                    # Case (ii)
            elif len(self.recency_cache) + len(self.ghost_recency_cache) < self.cache_size and (
                    len(self.recency_cache) + len(self.ghost_recency_cache) + len(self.frequency_cache) + len(
                self.ghost_frequency_cache)) >= self.cache_size:
                # "in Case IV(ii)"
                if (len(self.recency_cache) + len(self.ghost_recency_cache) + len(self.frequency_cache) + len(
                        self.ghost_frequency_cache)) == 2 * self.cache_size:
                    self.ghost_frequency_cache.pop()
                self.replace(item)

            self.recency_cache.appendleft(item)
            self.cache.add(item)
            self.miss += 1
        else:
            "There is an error."
            exit()

    def run_algorithm(self, blocktrace: list):
        for block in tqdm(blocktrace, disable=True):
            self.re(block)
        return self.hit / (self.hit + self.miss)

    def header(self):
        return (type(self).__name__)


class CacheHelper(abc.ABC):
    @abc.abstractmethod
    def replace(self, block):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def reference_cache(self, cache):
        pass

    @abc.abstractmethod
    def hitrate(self):
        pass

    @abc.abstractmethod
    def get_cache(self):
        pass

    @abc.abstractmethod
    def dereference_cache(self):
        pass


class LRU_Helper((CacheHelper)):
    def __init__(self, cache_size=10):
        self.cache_size = cache_size
        self.cache = set()
        self.recency = deque()
        self.hit = 0
        self.miss = 0

    def replace(self, block):
        if block in self.cache:
            self.recency.remove(block)
            self.recency.append(block)
            self.hit += 1
            return 1, 0

        elif len(self.cache) < self.cache_size:
            self.cache.add(block)
            self.recency.append(block)
            self.miss += 1
            return 0, 1
        else:
            self.cache.remove(self.recency[0])
            self.recency.popleft()
            self.cache.add(block)
            self.recency.append(block)
            self.miss += 1
            return 0, 1

    def hitrate(self):
        return self.hit / (self.hit + self.miss) if self.miss > 0 else 0

    def clear(self):
        self.hit = 0
        self.miss = 0

    def reference_cache(self, cache):
        self.cache = cache

    def get_cache(self):
        return self.cache

    def dereference_cache(self):
        self.cache = self.cache.copy()

    def update_recency(self, recency):
        self.recency = recency.copy()

    def get_recency(self):
        return self.recency


class LFU_Helper(CacheHelper):

    def __init__(self, cache_size=10):
        self.cache_size = cache_size
        self.cache = set()
        self.cache_frequency = defaultdict(int)
        self.frequency = defaultdict(int)
        self.hit = 0
        self.miss = 0
        self.recency = deque()

    def replace(self, block):
        self.frequency[block] += 1
        if block in self.cache:
            self.hit += 1
            self.cache_frequency[block] += 1
            self.recency.remove(block)
            self.recency.append(block)
            return 1, 0

        elif len(self.cache) < self.cache_size:
            self.cache.add(block)
            self.cache_frequency[block] += 1
            self.recency.append(block)
            self.miss += 1
            return 0, 1

        else:
            e, f = min(self.cache_frequency.items(), key=lambda a: a[1])
            self.cache_frequency.pop(e)
            self.cache.remove(e)
            self.cache.add(block)
            self.recency.remove(e)
            self.recency.append(block)
            self.cache_frequency[block] = self.frequency[block]
            self.miss += 1
            return 0, 1

    def hitrate(self):
        return self.hit / (self.hit + self.miss) if self.miss > 0 else 0

    def clear(self):
        self.hit = 0
        self.miss = 0

    def reference_cache(self, cache):
        self.cache = cache

    def get_cache(self):
        return self.cache

    def dereference_cache(self):
        self.cache = self.cache.copy()

    def update_frequency(self):
        self.cache_frequency = dict()
        for block in self.cache:
            self.cache_frequency[block] = self.frequency[block]

    def update_recency(self, recency):
        self.recency = recency.copy()

    def get_recency(self):
        return self.recency


'''
Prioritize LFU
'''


class MY_OPT(CacheAlgorithm):
    def __init__(self, cache_size=10, swith_interval=10):
        self.cache_size = cache_size
        self.swith_interval = cache_size * swith_interval

    def run_algorithm(self, blocktrace):
        lru_helper = LRU_Helper(self.cache_size)
        lfu_helper = LFU_Helper(self.cache_size)
        current_cache = lfu_helper.get_cache()

        hit, miss = 0, 0
        current_policy = 1
        switch_counter = 0

        for i, block in enumerate(tqdm(blocktrace, disable=True)):
            if i % self.swith_interval == 0:
                lru_hitrate = lru_helper.hitrate()
                lfu_hitrate = lfu_helper.hitrate()
                if (lfu_hitrate > lru_hitrate and current_policy == 0):  # if now is lru, but lfu performs better
                    current_policy = 1 - current_policy
                    lfu_helper.reference_cache(current_cache)  # lru now
                    current_cache = lfu_helper.get_cache()  # switch to lfu
                    lru_helper.dereference_cache()
                    lru_helper.clear()
                    lfu_helper.clear()
                    lfu_helper.update_frequency()
                    lfu_helper.update_recency(lru_helper.get_recency())
                    switch_counter += 1

                if (lfu_hitrate < lru_hitrate and current_policy == 1):  # if now is lfu, but lru performs better
                    current_policy = 1 - current_policy
                    lru_helper.reference_cache(current_cache)  # lfu now
                    current_cache = lru_helper.get_cache()  # switch to lru
                    lfu_helper.dereference_cache()
                    lru_helper.clear()
                    lfu_helper.clear()
                    lru_helper.update_recency(lfu_helper.get_recency())
                    switch_counter += 1

            lru_hit, lru_miss = lru_helper.replace(block)
            lfu_hit, lfu_miss = lfu_helper.replace(block)
            if current_policy == 0:
                hit += lru_hit
                miss += lru_miss
            else:
                hit += lfu_hit
                miss += lfu_miss

        hitrate = hit / (hit + miss)
        print("switch counter" + str(switch_counter))
        return hitrate

    def header(self):
        return (type(self).__name__)


if __name__ == "__main__":
    '''
    Load Data
    '''
    input_file = './DATA/cheetah.cs.fiu.edu-110108-113008.2.blkparse'
    print(str(datetime.datetime.now()) + ":Loading Blocktrace")
    df = pd.read_csv(input_file, sep=' ', header=None)
    df.columns = ['timestamp', 'pid', 'pname', 'blockNo', \
                  'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
    print(str(datetime.datetime.now()) + ":Starting simulator...")

    '''
    Setup config
    '''
    blocktrace = df['blockNo'].tolist()
    trainBlockTrace_timetamp = df['timestamp'].tolist()
    # blocktrace = blocktrace[:int(len(blocktrace)*0.1)]
    workload_size = len(set(blocktrace))
    cache_size_array = np.asarray([20,60])
    algorithms = list()
    algorithms.append('LRU(cache_size)')
    # algorithms.append('MRU(cache_size)')
    # algorithms.append('CLOCK(cache_size,3)')
    algorithms.append('LFU(cache_size)')
    algorithms.append('OPT(cache_size)')
    algorithms.append('ARC(cache_size)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=1)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=2)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=4)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=8)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=10)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=16)')
    # algorithms.append('LeCar_Opt(cache_size,history_size_factor=1)')
    # algorithms.append('LeCar_Opt2(cache_size,history_size_factor=1)')
    # algorithms.append('(CLOCK_LFU(cache_size))')
    # algorithms.append('LeCar_Opt3(cache_size,history_size_factor=1)')
    #algorithms.append('LeCar_Opt4(cache_size,history_size_factor=1)')
    algorithms.append('LeCar_Opt5(cache_size,history_size_factor=1)')
    # algorithms.append('MY_OPT(cache_size,swith_interval=1)')

    print("********************************************************************************************************")
    print("*")
    print("*    Inpu file: " + input_file)
    print("*    Cache size: " + str(cache_size_array))
    print("*    Workload size: " + str(workload_size))
    print("*    Data size: " + str(len(blocktrace)))
    print("*    Cache size/workload ratio: " + str(np.round(cache_size_array / workload_size, 10)))
    print("*    Algorithms: " + str(algorithms))
    print("*")
    print("********************************************************************************************************")

    '''
    Run algorithm
    '''
    print()
    print(str(datetime.datetime.now()) + ":Run algorithms...")
    print()

    for cache_size in cache_size_array:
        print(" Cache Size:" + str(cache_size))
        for algo in algorithms:
            start_time = datetime.datetime.now()
            print("     " + eval(algo + '.header()') + ":" + str(
                round(eval(algo + '.run_algorithm(blocktrace)'), 10)) + " time:" + str(
                datetime.datetime.now() - start_time))
        print()
        print()

    print(str(datetime.datetime.now()) + ":done.")
