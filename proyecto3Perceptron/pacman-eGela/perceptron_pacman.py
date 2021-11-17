# coding=utf-8
# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState
import pdb



PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())

        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            clasepredicha=0
            for i in range(len(trainingData)):
                #pdb.set_trace()  # esto es un break point para que puedas comprobar el formato de los datos
                max = -10000000
                for j in trainingData[i][1]:
                    score=trainingData[i][0][j]  * self.weights #weights: {'foodcount':x} (x: balioa)
                    if(score > max):
                        max= score
                        clasepredicha= j

                if (clasepredicha != trainingLabels[i]):
                    # recalcular pesos
                    self.weights= self.weights + trainingData[i][0][trainingLabels[i]] #trainingData[i][0] legalmoves bakoitzak duen pisua (hiztegia)
                    #aurreko for-eko trainingData[i][0][j]-ren berdina
                    self.weights = self.weights - trainingData[i][0][clasepredicha] #iragarri dugun klasea ez denez egokia, pisuak kendu
