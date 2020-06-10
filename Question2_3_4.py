#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
import random
import glob
import pprint
import glob
import re
from pprint import pprint
import io
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

########################################    
def GenerateKeyValueDictionaryForFile(filename):
    
    fileHandle = open(filename, 'r')
    
    word_dict = dict()
    
    words = fileHandle.read()
    
    wordsList = words.split(',')
    
    for word in wordsList:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    
    word_dict['__totalWords__'] = len(wordsList)
            
    try:
        fileHandle.close()
    except IOError:
        print('Error when closing the file')
    
    return word_dict

########################################    
def CalculateIDFScore(word, documentCount, masterDict):
    
    numberOfOccurence = 0.0
    
    for key, value in enumerate(masterDict):
        if word in masterDict[value]:
            numberOfOccurence += 1.0

    IDFScore = math.log((documentCount) / numberOfOccurence)
    
    return IDFScore    
    
########################################        
def printTFIDFScores(fileHandle, word, TFScore, IDFScore, TFIDFScore):
    message = "Word is "+word+" , TF-Score is "+ str(TFScore)+" , IDF-Score is "+str(IDFScore)+ " , TFIDF Score is "+str(TFIDFScore)+"\n"
    fileHandle.write(message)
    return
    
########################################        
def generateDictionaryForEachFile(filePath):
    
    filename = filePath.split('\\')[-1]
    
    tempDictionary = dict()
    
    tempDictionary[filename] = GenerateKeyValueDictionaryForFile(filePath)
    
    return tempDictionary

########################################        
def sendForCalculation(dictionary, masterDictionary, folderName):
    
    for key,value in enumerate(dictionary):
        
        fileHandle = open('20_newsgroups\\'+folderName+'\\TFIDF_ScoresPerFile\\'+value, 'w') # createFileHandleForScoreWriting(value, folderName)
        
        for subKey, subValue in enumerate(dictionary[value]):
		
            if subValue is not '__totalWords__':

                TFScore = (float(dictionary[value][subValue])) / (float(dictionary[value]['__totalWords__']))
                IDFScore = CalculateIDFScore(subValue, len(dictionary.keys()), masterDictionary)
                score = (TFScore * IDFScore)
                printTFIDFScores(fileHandle, subValue, TFScore, IDFScore, score)

        try:
            fileHandle.close()
        except IOError:
            print('An error occured while closing')
    return

# Will only be called by the root worker
# Splits the file array and assigns them to each worker
#######################################################
def main(file_array, folderName):
    
    total_files = len(file_array)
    
    split_size = int(total_files / size)
    
    masterDictionary = dict()
    
    for worker in range(1,size):
        if ( (total_files % size) > 0 and worker == size-1):
            comm.send(file_array[worker * split_size : ], dest = worker, tag=0)

        else:
            comm.send(file_array[worker * split_size : (worker+1) * split_size], dest = worker, tag=0)
        
    partial_data = processData(file_array[0:split_size])
    masterDictionary.update(partial_data)
           
    for worker in range(1,size):
        data = comm.recv(source=worker, tag=0) 
        masterDictionary.update(data)
        
    for worker in range(1,size):
        comm.send(masterDictionary, dest=worker, tag=3)
    
    sendForCalculation(partial_data, masterDictionary, folderName)                
     
# Called by all workers
#######################################################
def Worker(folderName):

    data = comm.recv(source=0, tag = 0)
    
    partial_dict = processData(data)
    
    comm.send(partial_dict, dest=0, tag=0)
    
    #########
    
    masterDictionary = comm.recv(source=0, tag=3)
    
    sendForCalculation(partial_dict, masterDictionary, folderName)
    
#######################################################
def processData(fileArray):

    partial_dict = dict()
    
    for i in range(len(fileArray)):
        fileDict = generateDictionaryForEachFile(fileArray[i])
        
        partial_dict.update(fileDict)
        
    return partial_dict        
            
#######################################################
if __name__=='__main__':
    
	startTime = MPI.Wtime()
	
	names = []
	
	for name in glob.glob('20_newsgroups\\*'):   # get all the subfolders
		names.append(name.split('\\')[-1])
    
	print(names)
    
	for number in range(0,5):
		if rank == 0:
			fileArray = (glob.glob('20_newsgroups\\'+names[number]+'\\cleansed_files\\*'))
			print(names[number])
			main(fileArray, names[number])
		else:
			Worker(names[number])    
        
		print('Done')
		
		comm.Barrier()
		
		if rank == 0:
			print("\nTime taken to process with " + str(size) + " workers is : " + str(MPI.Wtime() - startTime))
