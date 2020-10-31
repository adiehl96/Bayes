"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""
from read_bayesnet import BayesNet
from variable_elim import VariableElimination
import time


if __name__ == '__main__':
    lel = time.time()
    # The class BayesNet represents a Bayesian network from a .bif file in several variables
    believeNetwork = './networks/earthquake.bif'
    net = BayesNet(believeNetwork) # Format and other networks can be found on http://www.bnlearn.com/bnrepository/
    
    # Make your variable elimination code in a seperate file: 'variable_elim'. 
    # You can call this file as follows:
    ve = VariableElimination(net, believeNetwork)

    # Set the node to be queried as follows:
    query = 'JohnCalls'

    # The evidence is represented in the following way (can also be empty when there is no evidence):
    #evidence = {'alcoholism':'present', 'age': 'age65_100', 'PBC':'present'}
    evidence = {'Burglary':'True', 'Earthquake': 'True'}
    # Determine your elimination ordering before you call the run function. The elimination ordering   
    # is either specified by a list or a heuristic function that determines the elimination ordering
    # given the network. Experimentation with different heuristics will earn bonus points. The elimination
    # ordering can for example be set as follows:
    elim_order = net.nodes
    

    # Call the variable elimination function for the queried node given the evidence and the elimination ordering as follows:
    
    ve.run(query, evidence, elim_order)
    print(time.time()-lel)