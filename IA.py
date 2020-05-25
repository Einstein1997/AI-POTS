from typing import List, Any

import numpy,os
from random import random, randint
import numpy as np
import output as output
from scipy.stats import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def config():
    file1 = open("config_info","w")
    nom = input("nom de la plante :")
    age = input("age de la plante :")
    file1.write(nom + "\n")
    file1.write(age)
    file1.close()



file = open("config", "r+")
if "1" in file:
    print("cofigure")
else:
    config()
    file.write("1")
file.close()

print("Voulez vous enregistrer un autre arbre?")
enrg=input()
if enrg =="Non":
    print("souhaitez vous continuer?")
    rpns=input()
    print("ok")
    print("-------------------------")


Temperature = []
T = np.array(Temperature)
sommeT=[]
#Matrice des temprératures;
for i in range(18):
    Temperature.append(randint(20,39)+random())
for i in range(len(Temperature)):
    v= sum (Temperature)
print("Matrice Temprérature:  ")
print(Temperature)
print(sommeT)
#Matrice d'humidité;
humide=[]
H=np.array(humide)
sommeH=[]
for i in range(18):
    humide.append(randint(0, 99) + random())
for i in range(len(humide)):
    sommeH = sum(humide)
print("Matrice Humidité:  ")
print(humide)
print(sommeH)
#Matirce normal Temprérature.
MT=[]
for i in range(18):
   MT.append(25)
print("les valeurs normales de temprératures sont :")
print(MT)
moyT = sum ( MT ) /len ( MT)
print(moyT)
#Matirce normal Humidité.
MH=[]
for i in range(18):
    MH.append(50)
print("les valeurs normales d'humidité sont :")
print(MH)
moyH = sum ( MH ) /len ( MH)
print(moyH)

models = []
models.append(('LDA', LinearDiscriminantAnalysis(Temperature,humide)))

print("Show the model:")
print("******************")
print(models)
print("******************")



