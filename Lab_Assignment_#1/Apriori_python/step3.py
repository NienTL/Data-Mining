from csv import reader
from collections import defaultdict
from itertools import chain, combinations

from collections import defaultdict, OrderedDict
from csv import reader
from itertools import chain, combinations
from optparse import OptionParser

import os
import time

class Node:
    def __init__(self, itemName, frequency, parentNode):
        self.itemName = itemName
        self.count = frequency
        self.parent = parentNode
        self.children = {}
        self.next = None

    def increment(self, frequency):
        self.count += frequency

    def display(self, ind=1):
        print('  ' * ind, self.itemName, ' ', self.count)
        for child in list(self.children.values()):
            child.display(ind+1)

def getFromFile(fname):
    """Read tranction in the .csv file"""
    itemSetList = []
    frequency = []
    
    with open(fname, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            itemSetList.append(line)
            frequency.append(1)

    return itemSetList, frequency

def constructTree(itemSetList, frequency, minSup):
    """Construct the FP-tree and tje header table"""
    headerTable = defaultdict(int)
    # Counting frequency and create header table
    for idx, itemSet in enumerate(itemSetList):
        for item in itemSet:
            headerTable[item] += frequency[idx]

    # Deleting items below minSup
    headerTable = dict((item, sup) for item, sup in headerTable.items() if sup >= minSup)
    if(len(headerTable) == 0):
        return None, None

    # HeaderTable column [Item: [frequency, headNode]]
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]

    # Init Null head node
    fpTree = Node('Null', 1, None)
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate(itemSetList):
        itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])

    return fpTree, headerTable

def updateHeaderTable(item, targetNode, headerTable):
    if(headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode

def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].increment(frequency)
    else:
        # Create a new branch
        newItemNode = Node(item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable(item, newItemNode, headerTable)

    return treeNode.children[item]

def ascendFPtree(node, prefixPath):
    """Ascend the FP-tree to find the prefix path"""
    if node.parent != None:
        prefixPath.append(node.itemName)
        ascendFPtree(node.parent, prefixPath)

def findPrefixPath(basePat, headerTable):
    """Find all prefix path for a given base pattern."""
    # First node in linked list
    treeNode = headerTable[basePat][1] 
    condPats = []
    frequency = []
    while treeNode != None:
        prefixPath = []
        # From leaf node all the way to root
        ascendFPtree(treeNode, prefixPath)  
        if len(prefixPath) > 1:
            # Storing the prefix path and it's corresponding count
            condPats.append(prefixPath[1:])
            frequency.append(treeNode.count)

        # Go to next node
        treeNode = treeNode.next  
    return condPats, frequency

def mineTree(headerTable, minSup, preFix, freqItemList):
    """Mine the FP-tree to find all frequent itemsets"""
    # Sort the items with frequency and create a list
    sortedItemList = [item[0] for item in sorted(list(headerTable.items()), key=lambda p:p[1][0])] 
    # Start with the lowest frequency
    candidate_count_before_pruning = 0
    candidate_count_after_pruning = 0
    for item in sortedItemList: 
        candidate_count_before_pruning +=1
        # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
        newFreqSet = preFix.copy()
        newFreqSet.add(item)
        freqItemList.add(frozenset(newFreqSet))
        # Find all prefix path, constrcut conditional pattern base
        conditionalPattBase, frequency = findPrefixPath(item, headerTable) 
        # Construct conditonal FP Tree with conditional pattern base
        conditionalTree, newHeaderTable = constructTree(conditionalPattBase, frequency, minSup) 
        if newHeaderTable != None:
            # Mining recursively on the tree
            mineTree(newHeaderTable, minSup, newFreqSet, freqItemList)
        candidate_count_after_pruning += 1 
    return candidate_count_before_pruning, candidate_count_after_pruning

def powerset(s):
    """Generate all non-empty subsets of a set"""
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def getSupport(testSet, itemSetList):
    count = 0
    for itemSet in itemSetList:
        if(set(testSet).issubset(itemSet)):
            count += 1
    return count

def associationRule(freqItemSet, itemSetList, minConf):
    """Generate association rules from the frequent itemsets"""
    rules = []
    for itemSet in freqItemSet:
        subsets = powerset(itemSet)
        itemSetSup = getSupport(itemSet, itemSetList)
        for s in subsets:
            confidence = float(itemSetSup / getSupport(s, itemSetList))
            if(confidence > minConf):
                rules.append([set(s), set(itemSet.difference(s)), confidence])
    return rules

def getFrequencyFromList(itemSetList):
    frequency = [1 for i in range(len(itemSetList))]
    return frequency

def fpgrowth(itemSetList, minSupRatio, minConf):
    """Run FP-growth algorithm"""
    frequency = getFrequencyFromList(itemSetList)
    minSup = len(itemSetList) * minSupRatio
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    if(fpTree == None):
        print('No frequent item set')
    else:
        freqItems = set()
        mineTree(headerTable, minSup, set(), freqItems)
        rules = associationRule(freqItems, itemSetList, minConf)
        return freqItems, rules

def fpgrowthFromFile(fname, minSupRatio, minConf):
    itemSetList, frequency = getFromFile(fname)
    minSup = len(itemSetList) * minSupRatio
    # minSup = minSupRatio
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    if(fpTree == None):
        print('No frequent item set')
    else:
        candidate_counts = []
        freqItems = set()
        candidate_count_before_pruning, candidate_count_after_pruningmineTree = mineTree(headerTable, minSup, set(), freqItems)
        candidate_counts.append((candidate_count_before_pruning, candidate_count_after_pruningmineTree))
        rules = associationRule(freqItems, itemSetList, minConf)

        return freqItems, rules, candidate_counts, itemSetList

def printResults(freqItems, candidate_counts,dataset_name, outputPath, task_num, support_str, transactionList):
    result1_filename = os.path.join(outputPath, f"step3_task{task_num}_{dataset_name}_{support_str}_result1.txt")
    result2_filename = os.path.join(outputPath, f"step3_task{task_num}_{dataset_name}_{support_str}_result2.txt")

    with open(result1_filename, 'w') as f1:
        total_transactions = len(transactionList)
        itemSetSupportList = []
        for itemSet in freqItems:
            support = getSupport(itemSet, transactionList) / total_transactions
            if support >= float(support_str):
                itemSetSupportList.append((itemSet, support))
        sorted_itemsets = sorted(itemSetSupportList, key=lambda x: x[1], reverse=True)

        for itemSet, support in sorted_itemsets:
            support = support * 100
            item_str = "{" + ",".join(itemSet) + "}"
            f1.write(f"{support:.1f}\t{item_str}\n")

    with open(result2_filename, 'w') as f2:
        f2.write(f"{len(freqItems)}\n")
        for idx, (before, after) in enumerate(candidate_counts, 1):
            f2.write(f"{idx}\t{before}\t{after}\n")

def convert_data_to_csv(data_file):
    csv_file = data_file.replace(".data", ".csv")
    with open(data_file, 'r') as infile, open(csv_file, 'w') as outfile:
        for line in infile:
            line = line.strip().replace(" ", ",")
            outfile.write(line + "\n")
    return csv_file

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile', dest='input', help='filename containing csv or data',default='datasetA.data')
    optparser.add_option('-s', '--minSupport', dest='minS', help='minimum support value', default=0.01, type='float')
    optparser.add_option('-c', '--minConfidence', dest='minC', help='Min confidence (float)', default=0.5, type='float')
    optparser.add_option("-p", "--outputPath", dest="output", help="path to save output files", default='./')

    (options, args) = optparser.parse_args()

    input_file = options.input
    str_inputFile = input_file.split('.')[0]
    minSupport = options.minS
    minConfidence = options.minC
    output_path = options.output

    # Check and build the output file path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if input_file.endswith(".data"):
        # Convert .data file to .csv file
        input_file = convert_data_to_csv(input_file)

    start_task1 = time.time()
    freqItemSet, rules, candidate_counts, tranctionList = fpgrowthFromFile(input_file, minSupport, minConfidence)
    printResults(freqItemSet, candidate_counts, str_inputFile, output_path, 1, str(minSupport), tranctionList)
    end_task1 = time.time()
    time_task1 = end_task1 - start_task1
    print(f"Task 1 computation time: {time_task1:.2f} seconds.")
    # print(freqItemSet)
    # print(rules)