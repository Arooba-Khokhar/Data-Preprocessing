#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
import random
import glob
from matplotlib import pyplot as plt
import cleaning_file as utility_functions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Will only be called by the root worker
# Splits the file array and assigns them to each worker
##############################################        
def main(file_array, folderName):
    
    total_files = len(file_array)
    
    split_size = int(total_files / size)
    
    for worker in range(1,size):
        if ( (total_files % size) > 0 and worker == size-1):
            comm.send(file_array[worker * split_size : ], dest = worker, tag=0)

        else:
            comm.send(file_array[worker * split_size : (worker+1) * split_size], dest = worker, tag=0)
    
    utility_functions.readFromFileAndGenerateTokens(file_array[0:split_size], folderName)
    
    
# Called by all workers
##############################################        
def Worker(folderName):

    file_Array = comm.recv(source=0, tag = 0)
    
    utility_functions.readFromFileAndGenerateTokens(file_Array, folderName)    
    
##############################################        
if __name__=='__main__':
    
    start = MPI.Wtime()
	
    names = []
    values = []
        
    for name in glob.glob('20_newsgroups\\*'):   # get all the subfolders
        names.append(name.split('\\')[-1])
    
    for name in names:
        
        if rank == 0:
            fileArray = (glob.glob('20_newsgroups\\'+name+'\\*'))
            #print(name)
            main(fileArray, name)
        else:
            Worker(name)    
        
        print('Done with folder {}'.format(name))    
        #print("\nTime taken to process with " + str(size) + " workers is : " + str(MPI.Wtime() - start))
        values.append((MPI.Wtime() - start))
        comm.Barrier()
        
    if rank == 0:
        print("\nTime taken to process with " + str(size) + " workers is : " + str(MPI.Wtime() - start))
