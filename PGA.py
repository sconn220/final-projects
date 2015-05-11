# -*- coding: utf-8 -*-
"""
Partition Genetic Algorithm -
"""
import numpy as np
import operator
from copy import deepcopy
from deap import creator,base,tools,algorithms
import matplotlib.pyplot as plt
toolbox = base.Toolbox()

class Organism(object):
    '''
    An organism is defined by its partition and has resulting fitness.
    Fitness is determined by the enviroment it is in.
    '''
    def __init__(self, partition):
        self.partition = partition
        self.fitness = None
    

   
class Population(object):
    """
    Contains all the organism for one generations
    """
    def __init__(self, population, nprtl):
        self.population_size = len(population)
        self.nprtl = nprtl
        self.organisms = population 
        self.min_fitness = self.__Min_Fitness()
        
        
    def __Min_Fitness(self):
        """
        Calculates Min Fitness of that generation of the population    
        """
        min_fitness = self.organisms[0].fitness    
        for ii in range(self.population_size):
            if (self.organisms[ii].fitness != None):
                if (self.organisms[ii].fitness < min_fitness):
                    min_fitness = self.organisms[ii].fitness
        return min_fitness
        
class Environment(object):
    def __init__(self,weights,nprtl,pop_size,nbins):
        self.nprtl = nprtl
        self.nbins = nbins
        self.population_size = pop_size
        self.weights = weights
        self.generation = self.__initial_population()
        self.num_generations = 0
        
    def __initial_population(self):
        '''Creates an initial population of organisms'''
        population = []
        for ii in range(self.population_size):
            partition = np.random.randint(self.nbins, size = self.nprtl)            
            organism = Organism(partition)
            organism.fitness = Fitness_Multi_Bin(organism,self.nbins,self.weights)
            population.append(organism)
        generation = Population(population,self.nprtl)
        return [generation]
              
    def Create_New_Generation(self):
        '''
        Create New Generation by performing the following steps
                1) orders the population from min fitness to max fitness
                2) Selects parents from ordered list to 'Mate'
                3) assigns fitness to new organism and repeat 
        '''
        self.num_generations +=1
        g_idx = self.num_generations - 1
        current_gen = deepcopy(self.generation[g_idx])
        regen = order_generation(current_gen)
        new_pop = []
        for ii in range(int(self.generation[g_idx].population_size/2)):
            #Select Parents
#            p_sel = Roulette_Wheel_Mate_Selection(self.generation[g_idx].population_size)           
            p_sel = Boltzmann_Tournament_Selection(regen)
            #Mate Parents            
            new_org1, new_org2 = Mating_Single_Cross_Over(regen.organisms[p_sel[0]],regen.organisms[p_sel[1]])
            
            #Mutation
            Mutation_Organism(new_org1,self.nprtl,self.nbins)
            Mutation_Organism(new_org2,self.nprtl,self.nbins)
            
            #Set Children's fitness            
            new_org1.fitness = Fitness_Multi_Bin(new_org1,self.nbins,self.weights)
            new_org2.fitness = Fitness_Multi_Bin(new_org2,self.nbins,self.weights)
            
            #Add to new Generations population            
            new_pop.append(new_org1)
            new_pop.append(new_org2)
            
        #Create new Population object and append it to the generation class Variable
        new_gen = Population(new_pop,self.nprtl)
        self.generation.append(new_gen)
        
def order_generation(current_gen):
    '''Orders Population from min to max fitness'''
    current_gen.organisms.sort(key = lambda x: x.fitness)
    return current_gen

def Boltzmann_Tournament_Selection(pop):
    B = 1
    T = 100000    
    prob = np.zeros(pop.population_size)
    boltzman = np.zeros(pop.population_size)
    for ii in range(pop.population_size):
        boltzman[ii] = np.exp(-B*pop.organisms[ii].fitness/T)
    
    #find probability
    for ii in range(pop.population_size):
        prob[ii] = boltzman[ii]/sum(boltzman)
    probsum = np.cumsum(prob)
    rand1 = np.random.rand(1)
    rand2 = np.random.rand(1)
    parent1 = np.digitize(rand1,probsum)
    parent2 = np.digitize(rand2,probsum)
    if parent1 == parent2:
        parent2 = parent2 + 1
    return [parent1,parent2]
    

def Roulette_Wheel_Mate_Selection(pop_size):
    parents = np.arange(pop_size)+1
    prob = np.zeros(pop_size)
    for ii in range(pop_size):
        prob[ii] = parents[ii]/np.sum(parents)
    prob[::-1] = prob #niffty way to reverse array
    probsum = np.cumsum(prob)
    rand1 = np.random.rand(1)
    rand2 = np.random.rand(1)
    parent1 = np.digitize(rand1,probsum)
    parent2 = np.digitize(rand2,probsum)
    if parent1 == parent2:
        parent2 = parent2 + 1
    return [parent1,parent2]
                
def Natrual_Selection(population, NUM2KILL):
    '''
    "kills" the lowest N performing organisms by making their
    fitness and partition None    
    '''
    population.organism.sort(key=operator.attrgetter('fitness'))
    for ii in range(NUM2KILL):
        population.organism[-ii-1].fitness = None
        population.organism[-ii-1].partition = None

def Mating(population):
    '''
    Mates two 'successful' genes by taking the first half of genes from one parent
    and the second half from the second parent
    '''
    parent1 = 1
    parent2 = 2
    if (population.population_size % 2 == 0):#Even Population Size
        lower_bounds = population.population_size/2
        upper_bounds = population.population_size
    else: #odd population size
        lower_bounds = (population.popuilation_size + 1)/2
        upper_bounds = population.population_size         
    for ii in range(population.population_size):
        if population.organism[ii].fitness == None:
            population.organism[ii].partition = population.organism[parent1].partition
            population.organism[ii].partition[lower_bounds:upper_bounds] = population.organism[parent2].partition[lower_bounds:upper_bounds]
            parent1 = parent1 + 1
            parent2 = parent2 + 1
            
def Mating_Single_Cross_Over(parent1,parent2):
    '''
    Generates offspring using a single cross over point.
    Make Selection comes from a roulette wheel approach
    '''
    # Generate Cross Over Point
    cross_over_point = np.random.randint(len(parent1.partition))
    
    #Deep Copy Parents
    partition1 = deepcopy(parent1.partition)
    partition2 = deepcopy(parent2.partition)
    
    #Deep Copy Opposite parent 'Chunk' up to cross over point
    partition1[0:cross_over_point] = deepcopy(parent2.partition[0:cross_over_point])
    partition2[0:cross_over_point] = deepcopy(parent1.partition[0:cross_over_point])
    
    #Create Two new Child Organisms and return them 
    organism1 = Organism(partition1)
    organism2 = Organism(partition2)
    return organism1, organism2
        
def Mating_PMX_Crossover(parent1,parent2):
    '''
    Generates offspring using Partially Matched Crossover
    there are two cross over points. Then there is an ordered maping that 
    computes the rest of the offspring
    '''
    #generate cross over points
    cross_over_points = np.random.randint(len(parent1.partition), size = (2,1))
    if cross_over_points[0] == cross_over_points[1]:
        print('points where equal but now fixed')
        cross_over_points[1] -= 1
    cross_over_points = np.sort(cross_over_points, axis=None) #order 
    print('crossover points',cross_over_points)
    
    #cross over 
    offspring1 = deepcopy(parent1.partition)
    offspring2 = deepcopy(parent2.partition)
    print(offspring1)
    print(offspring2)
    offspring1[cross_over_points[0]:cross_over_points[1]] = deepcopy(parent2.partition[cross_over_points[0]:cross_over_points[1]])
    offspring2[cross_over_points[0]:cross_over_points[1]] = deepcopy(parent1.partition[cross_over_points[0]:cross_over_points[1]])
    print(offspring1)
    print(offspring2)
    return[offspring1,offspring2]

def Fitness_Multi_Bin(organism,NBINS,weights):
    bins = np.zeros(NBINS)
    for ii in range(len(organism.partition)):
        bin_num = organism.partition[ii]        
        bins[bin_num] =bins[bin_num] + weights[ii]
    idealsol = sum(weights)/NBINS
    return sum(abs(bins-idealsol))
    
    
    
def Mutation_Organism(organism,nprtl,nbins):
    '''
    Randomly will mutate a gene from a parents offspring before their fitness is tested
    '''
    if (organism.fitness == None)&(np.random.rand(1) > 0.95):
        rand_index = int(np.random.rand(1)*nprtl)
        organism.partition[rand_index] = np.random.randint(nbins)

#def Fitness(partition,weights):
#    '''Absolute value of the difference of the two bins is the fitness
#    of the oranism. The lower the fitness the more fit the organism'''
#    Bin1 = 0
#    Bin2 = 0
#    for jj in range(len(partition)):
#        if partition[jj] == 0:
#            Bin1 = Bin1 + weights[jj]
#        else:
#            Bin2 = Bin2 + weights[jj]
#    return abs(Bin1-Bin2)


NPRTL = 34
NBINS = 10
POPULATION_SIZE = 200
NGEN = 100
weights = [ 3380, 1824, 1481, 2060, 1225, 836, 1363, 2705, 4635, 648, 2588, 3380, 1952, 3832, 3176, 2316, 2479, 3433, 3519, 1363, 1824, 3305, 2156, 3305, 3049, 3980, 2787, 4635, 4068, 2992, 5932, 528, 3304, 4107]

print(weights)
test = Environment(weights,NPRTL,POPULATION_SIZE,NBINS)
#Boltzmann_Tournament_Selection(test.generation[0])
fitplot = np.zeros(NGEN)
for ii in range(NGEN):
    test.Create_New_Generation()
    fitplot[ii] = test.generation[ii].min_fitness
    print(ii,test.generation[ii].min_fitness)

plt.plot(fitplot)
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.show()



