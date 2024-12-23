import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import time
import multiprocessing as mp
import os

# 沒有用到
def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    # 平行計算min support
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(check_item_support, [(item, transactionList) for item in itemSet])
    
    for item, count in results:
        freqSet[item] += count
        support = float(count) / len(transactionList)
        # 把suport >= minSupport 的 itemset加入＿itemSet，並在最後回傳_itemSet
        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

def check_item_support(item, transactionList):
    count = sum(1 for transaction in transactionList if item.issubset(transaction))
    return item, count

def joinSet(itemSet, length):
    # 把itemset中的元素合併成lengeth-element itemsets
    # 合併之前檢查子及是否存在itemSet，確保頻繁的規則
    return set(
        [i.union(j) for i in itemSet for j in itemSet 
         if len(i.union(j)) == length and all(frozenset(subset) in itemSet for subset in combinations(i.union(j), length - 1))]
    )

def getItemSetTransactionList(data_iterator):
    # 生成tranction list
    transactionList = list()
    itemSet = set()
    
    for record in data_iterator:  
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            # 生成 1-element itemsets
            itemSet.add(frozenset([item]))  
            
    return itemSet, transactionList

def runApriori(data_iter, minSupport):
    # initial 1-element itemsets
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()
    # oneCSet: 找到符合minimum support 的Itemsets
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    # currentLSet: 儲存目前的frequent itemsets
    currentLSet = oneCSet
    k = 2
    candidate_counts = []

    while currentLSet != set([]):
        # 計算candidate before pruning的數量
        candidate_count_before_pruning = len(currentLSet)
        largeSet[k - 1] = currentLSet
        # k遞增，逐漸找到長度越來越長的frequent itemsets
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet, transactionList, minSupport, freqSet)
        # 計算candidate after pruning的數量
        candidate_count_after_pruning = len(currentCSet)
        candidate_counts.append((candidate_count_before_pruning, candidate_count_after_pruning))
        currentLSet = currentCSet
        k += 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    # 儲存每個frequent itemset的support
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems, candidate_counts, freqSet, largeSet, transactionList

def printResults(items, candidate_count, dataset_name, output_path, task_num, support_str):
    idx = 1
    total_num_itemset = len(items)
    result1_filename = os.path.join(output_path, f"step2_task{task_num}_{dataset_name}_{support_str}_result1.txt")
    result2_filename = os.path.join(output_path, f"step2_task{task_num}_{dataset_name}_{support_str}_result2.txt")
    
    with open(result1_filename, 'w') as f1:
        with open(result2_filename, 'w') as f2:
            f2.write(str(total_num_itemset) + "\n")
            for item, support in sorted(items, key=lambda x: x[1], reverse=True):
                item_str = "{" + ",".join(item) + "}"
                support = support * 100
                f1.write(f"{support:.1f}\t{item_str}\n")
            for candi in candidate_count:
                f2.write(f"{idx}\t{candidate_count[idx-1][0]}\t{candidate_count[idx-1][1]}\n")
                idx += 1

def dataFromFile(fname):
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")
            record = frozenset(line.split(","))
            # print("*** record: ", record)
            yield record

def isClosed(freqSet, item, support):
    """Check if an itemset is closed"""
    for other_item in freqSet:
        if item != other_item and item.issubset(other_item) and freqSet[other_item] == support:
            return False
    return True

def getFrequentClosedItemsets(freqSet, largeSet, transactionList):
    """Filter for frequent closed itemsets"""
    closed_itemsets = []
    
    for key, value in largeSet.items():
        for item in value:
            support = float(freqSet[item]) / len(transactionList)
            if isClosed(freqSet, item, freqSet[item]):
                closed_itemsets.append((tuple(item), support))
    
    return closed_itemsets

def printClosedItemsets(closed_items, dataset_name, output_path, support_str):
    result_filename = os.path.join(output_path, f"step2_task2_{dataset_name}_{support_str}_result1.txt")
    total_num_closed_itemsets = len(closed_items)
    with open(result_filename, 'w') as f:
        f.write(str(total_num_closed_itemsets) + "\n")
        for item, support in sorted(closed_items, key=lambda x: x[1], reverse=True):
            item_str = "{" + ",".join(item) + "}"
            support = support * 100
            f.write(f"{support:.1f}\t{item_str}\n")

def convert_data_to_csv(data_file):
    csv_file = data_file.replace(".data", ".csv")
    with open(data_file, 'r') as infile, open(csv_file, 'w') as outfile:
        for line in infile:
            line = line.strip().replace(" ", ",")
            outfile.write(line + "\n")
    return csv_file

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option("-f", "--inputFile", dest="input", help="filename containing csv or data", default='datasetA.data')
    optparser.add_option("-s", "--minSupport", dest="minS", help="minimum support value", default=0.01, type="float")
    optparser.add_option("-p", "--outputPath", dest="output", help="path to save output files", default='./')
    
    (options, args) = optparser.parse_args()

    input_file = options.input
    minSupport = options.minS
    output_path = options.output
    
    support_str = str(minSupport)
    str_inputFile = str(input_file)
    dataset_name = str_inputFile.split('.')[0]
    print("input_file: ", input_file)
    print("minSupport: ", minSupport)
    print("output_path: ", output_path)
    print("support_str", support_str)
    
    # Check and build the output file path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if input_file.endswith(".data"):
        # Convert .data file to .csv file
        input_file = convert_data_to_csv(input_file)

    inFile = dataFromFile(input_file)

    print(f"Minimum Support(%): {minSupport*100}")
    
    # Task 1
    start_task1 = time.time()
    items, candi, freqSet, largeSet, transactionList = runApriori(inFile, minSupport)
    printResults(items, candi, dataset_name,output_path, 1, support_str)
    end_task1 = time.time()
    time_task1 = end_task1 - start_task1

    # Task 2
    start_task2 = time.time()
    closed_items = getFrequentClosedItemsets(freqSet, largeSet, transactionList)
    printClosedItemsets(closed_items, dataset_name, output_path, support_str)
    end_task2 = time.time()
    # Task 2 的計算時間包含task 1 mine frequent itemsets 的時間
    time_task2 = end_task2 - start_task2 + time_task1 

    print(f"Task 1 computation time: {time_task1:.2f} seconds.")
    print(f"Task 2 computation time: {time_task2:.2f} seconds.")
    ratio = time_task2 / time_task1
    print(f"Time ratio (Task2/Task1): {ratio:.2f}%")