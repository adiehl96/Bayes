"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk, Arne Diehl

Class for the implementation of the variable elimination algorithm.

"""

import pandas as pd
import numpy  as np
from copy import deepcopy
import os

class VariableElimination():

    def __init__(self, network, name):
        """
        Initialize the variable elimination algorithm with the specified network.
        Add more initializations if necessary.

        """
        self.network = network
        self.file = None
        number=0
        while self.file is None:
            if not os.path.isfile('./log'+str(number)+'.txt'):
                self.file = open("log"+str(number)+".txt","a+")
            number += 1
        self.file.write('Log for VE on the network file: ' + name + '\n')
        
    def remove_barren(self):
        """
        
        Check for every existing nodes if it is a parent node for any other node.
        If not, and it is not the query or observed, remove it from further calculation.
        
        """
        index = 0
        self.relevant_nodes = self.network.nodes
        while index<len(self.relevant_nodes):
            candidate = self.relevant_nodes[index]
            count=0
            if (candidate==self.query) or (candidate in self.observed.keys()):
                index+=1
                continue
            for key, value in self.network.parents.items():
                for parent in value:
                    if parent == candidate:
                        count+=1;
            if(count==0):
                del(self.network.parents[candidate])
                del(self.network.probabilities[candidate])
                del(self.relevant_nodes[index])
                index = 0
                continue
            index+=1
            
        self.file.write('list of non-barren nodes: ' + str(self.relevant_nodes)+'\n\n')
        self.file.flush()
    
    def reduce_observed(self):
        """
        Loop through the evidence dictionary and inner Loop through the factor dictionary and search for factors using the observed variable.
        Set every usage in all childnodes to the value of the observed value.
        
        
        """
        
        for observation in self.observed.keys():
            newFactors = []
            for factorname in self.factors.keys():
                factor = self.factors[factorname]
                if list(factor.columns).count(observation)>0:
                    self.usedList += [factorname]
                    if len(list(factor.columns))>2:
                        # print(dir(factor[observation]))
                        values=set(factor[observation].values)
                        values.remove(self.observed[observation])
                        for value in values:
                            indizes = list(factor.groupby(observation).groups[value])
                            factor = factor.drop(indizes)
                        factor = factor.drop([observation],axis=1)
                        newFactors += [factor]
                        
    
            if newFactors is not []:
                for newFactor in newFactors:
                    newName = np.amax([np.amax(list(self.factors.keys())),np.amax(self.usedList)])+1
                    self.factors[newName] = newFactor
                    self.file.write('\nfactor nr: ' + str(newName) + '\n' + str(newFactor)+'\n')
                    self.file.flush()
                
            for used in self.usedList:
                keys = self.factors.keys()
                if used in keys:
                    del self.factors[used]
        
    def create_factor_dict(self):
        """
        
        this just translates the factors to a more useful naming convention
        
        """
        self.factors = {}
        for idx, (key, value) in enumerate(self.network.probabilities.items()):
            self.factors.update({idx: value})
            self.file.write('\nfactor nr: ' + str(idx) + '\n' + str(self.factors[idx])+'\n')
            self.file.flush()
        
    def sum_out(self, factor, parameter):
        """
        
        Check for every existing nodes if it is a parent node for any other node.
        If not, and it is not the query or observed, remove it from further calculation.
        
        """
        factor = factor.drop([parameter], axis=1)
        cols = factor.columns.tolist()
        cols.remove('prob')
        factor = factor.groupby(cols).agg('sum')
        factor = factor.reset_index()
        return factor
        
    def fac_mul(self, fac1, fac2):
        """
        
        Join two dataframes on the intersection of columns, create a new dataframe including all columns.
        Then sum the two value columns
        
        """
        fac1cols = fac1.columns.tolist()
        fac2cols = fac2.columns.tolist()

        intersect = list(set(fac1cols) & set(fac2cols))
        intersect.remove('prob')

        union = list(set(fac1cols) | set(fac2cols))
        union.remove('prob')

        merged = pd.merge(fac1, fac2, on=intersect).set_index(union)
        merged['prob'] = merged['prob_x'] * merged['prob_y']
        merged = merged.drop(['prob_x', 'prob_y'], axis=1)
        merged = merged.reset_index()
        return merged
    
    def eliminate_variable(self, variable):
        """
        
        Loop through the factor dictionary and search for factors using the variable.
        In case we find one, copy that factor to newFactor, add it's number to the usedList and continue searching.
        For every further factor we encounter perform the multiplication with the newFactor and store the result in newFactor,
        as well as add the factors number to the usedList. Afterwards remove all factors on the usedList from the dictionary,
        perform the sumOut operation on newFactor and add it to the dictionary.
        The new factores name is whatever number is highest in the dictionary or usedList
        
        """
        newFactor = None
        for factorname in self.factors.keys():
            factor = self.factors[factorname]
            if list(factor.columns).count(variable)>0:
                self.usedList += [factorname]
                if newFactor is None:
                    newFactor = factor
                else:
                    newFactor = self.fac_mul(factor, newFactor)

        if newFactor is not None and (len(list(newFactor.columns))>2):
            newName = np.amax([np.amax(list(self.factors.keys())),np.amax(self.usedList)])+1
            newFactor = self.sum_out(newFactor, variable)
            self.file.write('\nfactor nr: ' + str(newName) + '\n' + str(newFactor)+'\n')
            self.file.flush()
            self.factors[newName] = newFactor
        elif newFactor is not None and len(list(newFactor.columns))==2 and list(newFactor.columns).count(self.query)>0:
            newName = np.amax([np.amax(list(self.factors.keys())),np.amax(self.usedList)])+1
            self.file.write('\nfactor nr: ' + str(newName) + '\n' + str(newFactor)+'\n')
            self.file.flush()
            self.factors[newName] = newFactor
            
        for used in self.usedList:
            keys = self.factors.keys()
            if used in keys:
                del self.factors[used]
        
    def next_node(self):
        """
        
        In case we have an ordered elimination list, we take the next one, in the case we have a heuristic, compute the next elimination.
        
        """
        if self.elim_list is not None:
            while len(self.elim_list) > 0:
                node = self.elim_list.pop(0)
                if node in self.relevant_nodes and not node == self.query:
                    del(self.relevant_nodes[self.relevant_nodes.index(node)])
                    return node
            
        if self.heuristic is not None:
            return 'heuristic stuff' #not yet implemented
            
    def normalize(self, factor):
        factor['prob']=factor['prob']/(factor['prob'].sum())
        return factor

    
    def run(self, query, observed, elim_order):
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable

        """
        self.query = query
        self.file.write('query: ' + str(self.query) + '\n')
        self.observed = observed
        for observation in list(self.observed.keys()):
            self.file.write('observed variable: ' + str(observation) + '\n' + 'observed value: ' + self.observed[observation] + '\n')
        self.usedList = [-1]
        if type(elim_order) is list:
            self.elim_list = deepcopy(elim_order)
            self.file.write('Elimination order: ' + str(self.elim_list) + '\n')
        elif type(elim_order) is str:
            if elim_order == 'heuristic':
                self.heuristic = elim_order
            else:
                self.file.write('invalid elimination order parameter\n')
        else:
            self.file.write('invalid elimination order parameter\n')
        
        #now we remove the barren nodes
        self.remove_barren()
        
        #create the factors
        self.create_factor_dict()
        
        #reduce observed variables
        self.reduce_observed()
        
        #start the VE loop
        while len(self.relevant_nodes) > 1:
            next_elimination = self.next_node()
            self.file.write("next eliminiation: " + str(next_elimination) + '\n')
            self.eliminate_variable(next_elimination)
        """
        if len(list(self.factors.keys()))>1:
            newFactor = None
            for factorname in list(self.factors.keys()):
                factor = self.factors[factorname]
                if newFactor is None:
                    newFactor = factor
                else:
                    newFactor = self.fac_mul(factor, newFactor)
            newName = np.amax([np.amax(list(self.factors.keys())),np.amax(self.usedList)])+1
            
            self.eliminate_variable(self.query)
        """
        self.eliminate_variable(self.query)
        result = self.factors[list(self.factors.keys())[0]]
        result = self.normalize(result)
        
        print ('\n', result)
        self.file.write("\nResult: \n" + str(result) + '\n')
        self.file.flush()
        self.file.close
        