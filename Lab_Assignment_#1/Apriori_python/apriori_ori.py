"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport

    $python apriori.py -f DATASET.csv -s 0.15
"""

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

import time

######################### TASK 1 BELOW #########################
def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

# 生成1-element itemsets 和transaction列表
def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        # print("record: ", record) 
        # 將transation轉換成不可變集合
        transaction = frozenset(record)
        # print("transition: ", transaction)
        transactionList.append(transaction)
        for item in transaction:
            # item: 1-element itemset
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
        # print(itemSet)
            
    return itemSet, transactionList

def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """

    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport


    oneCSet= returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    print("oneCSet: ", oneCSet)
    currentLSet = oneCSet
    k = 2
    candidate_counts = []

    while currentLSet != set([]):
        candidate_count_before_pruning = len(currentLSet)
        # print("******** candidate_count_before_pruning", candidate_count_before_pruning, "********")

        largeSet[k - 1] = currentLSet
        # 不斷擴展itemsets 並計算support，直到無法生成更大的itemsets。
        currentLSet = joinSet(currentLSet, k)
        currentCSet= returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )

        candidate_count_after_pruning = len(currentCSet)
        # print("******** candidate_count_after_pruning", candidate_count_after_pruning, "********")
        candidate_counts.append((candidate_count_before_pruning, candidate_count_after_pruning))


        currentLSet = currentCSet
        k = k + 1
    

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems, candidate_counts


def printResults(items, candidate_count):
    """prints the generated itemsets sorted by support """
    # print("items: ", items)
    idx = 1;
    total_num_itemset = len(items);
    with open("A_0,3_task1_1_test.txt", 'w') as f1:
        with open("A_0.3_task1_2_test.txt", 'w') as f2:
            f2.write(str(total_num_itemset) + "\n")
            for item, support in sorted(items, key=lambda x: x[1], reverse=True):
                # print("item: %s , %.3f" % (str(item), support))
                item_str = "{" + ",".join(item) + "}"
                support = support * 100
                # print(f"{support:.1f}\t{item_str}\n")
                f1.write(f"{support:.1f}\t{item_str}\n")
            for candi in candidate_count:
                # print("candidata_count: ", candi[0], candi[1])
                # print("index: ", idx)
                f2.write(f"{idx}\t{candidate_count[idx-1][0]}\t{candidate_count[idx-1][1]}\n")
                idx += 1
        


    # print("the end of printResults!!!")


def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1], reverse=True):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)
    return i


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    count = 0
    with open(fname, "r") as file_iter:
        for line in file_iter:
            # print("***line: ", line)
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            count+=1
            # print(f"Read record: {record}")
            yield record
    # print(count)

######################### TASK 2 BELOW #########################
def isClosed(freqSet, item, support):
    """檢查該項集的所有超集，確保沒有超集具有相同的支持度"""
    for other_item in freqSet:
        if item != other_item and item.issubset(other_item) and freqSet[other_item] == support:
            return False
    return True

def getFrequentClosedItemsets(freqSet, largeSet, transactionList):
    """篩選出頻繁閉合項集"""
    closed_itemsets = []
    
    for key, value in largeSet.items():
        for item in value:
            support = float(freqSet[item]) / len(transactionList)
            if isClosed(freqSet, item, freqSet[item]):
                closed_itemsets.append((tuple(item), support))
    
    return closed_itemsets

def runAprioriForTask2(data_iter, minSupport):
    """專門為 Task 2 執行的 Apriori 演算法，篩選出頻繁閉合項集"""
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()
    
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    
    currentLSet = oneCSet
    k = 2

    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentLSet = returnItemsWithMinSupport(currentLSet, transactionList, minSupport, freqSet)
        k = k + 1

    closed_items = getFrequentClosedItemsets(freqSet, largeSet, transactionList)
    return closed_items

def printClosedItemsets(closed_items):
    """輸出頻繁閉合項集"""
    total_num_closed_itemsets = len(closed_items)
    with open("A_0,3_task2_test.txt", 'w') as f:
        f.write(str(total_num_closed_itemsets) + "\n")
        for item, support in sorted(closed_items, key=lambda x: x[1], reverse=True):
            item_str = "{" + ",".join(item) + "}"
            support = support * 100
            f.write(f"{support:.1f}\t{item_str}\n")
    # print("Closed Itemsets exported successfully!")


if __name__ == "__main__":
    # 從command line讀取parameters
    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default='A.csv'
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.1,
        type="float",
    )
    
    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
        # print("******** options.input is None!!! ********")
    elif options.input is not None:
        # print("******** option.input is not None!!! ********")
        inFile = dataFromFile(options.input)
        print("infile: ", inFile)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS
    print(" *********** minSupport: ", minSupport, "***********")

    # TASK 1
    start_task1 = time.time()
    items,candi = runApriori(inFile, minSupport)
    printResults(items, candi)
    # print("******* candidate_count: ", candi, "*******")
    end_task1 = time.time()
    time_task1 = end_task1 - start_task1
    print(f"Task1 computation time: {time_task1:.2f} seconds.")

    # 重新加載數據，確保 task2 獨立執行
    inFile = dataFromFile(options.input)
    
    # Task 2 獨立執行
    start_task2 = time.time()
    closed_items_task2 = runAprioriForTask2(inFile, minSupport)
    printClosedItemsets(closed_items_task2)
    end_task2 = time.time()
    time_task2 = end_task2 - start_task2
    print(f"Task2 computation time: {time_task2:.2f} seconds.")


    ratio = (time_task2 / time_task1) * 100
    print(f"Time ratio (Task2/Task1): {ratio:.2f}%")