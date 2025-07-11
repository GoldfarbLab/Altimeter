import pyopenms.Constants
import re
import csv
import numpy as np
import torch
from annotation import annotation
from dataset import createPeptide
import math


def calcIsotopeDistribution(seq, z, mods, annot_string, iso2efficiency):
    # parse annotation
    annot = annotation.from_entry(annot_string, z)
    pep = createPeptide(seq, mods)
    return annot.getTheoreticalIsotopeDistribution(pep, iso2efficiency)


    
class DicObj:
    def __init__(self,
                 stats_path,
                 mod_config,
                 seq_len = 40,
                 chlim = [1,8]
                 ):
        self.seq_len = seq_len
        self.chlim = chlim
        self.chrng = chlim[-1]-chlim[0]+1
        self.dic = {b:a for a,b in enumerate('ARNDCQEGHILKMFPSTWYVX')}
        self.revdic = {b:a for a,b in self.dic.items()}
        #self.mdic = {b:a+len(self.dic) for a,b in enumerate([
        #        '','Acetyl', 'Carbamidomethyl', 'Gln->pyro-Glu', 'Glu->pyro-Glu', 
        #        'Oxidation', 'Phospho', 'Pyro-carbamidomethyl', 'TMT6plex'])}
        self.mdic = {b:a+len(self.dic) for a,b in enumerate([mod for mod in (list(mod_config['var_mods'].keys()) + list(mod_config['fixed_mods'].keys()))])}
        self.revmdic = {b:a for a,b in self.mdic.items()}
        
        self.parseIonDictionary(stats_path)
        
        self.seq_channels = len(self.dic) + len(self.mdic)
        self.channels = len(self.dic) + len(self.mdic) + self.chrng + 1
        
        # Synonyms
        if 'Carbamidomethyl' in self.mdic.keys():
            self.mdic['CAM'] = self.mdic['Carbamidomethyl']
            self.revmdic[self.mdic['CAM']] = 'CAM'
        elif 'CAM' in self.mdic.keys():
            self.mdic['Carbamidomethyl'] = self.mdic['CAM']
            self.revmdic[self.mdic['Carbamidomethyl']] = 'Carbamidomethyl'
        if 'TMT6plex' in self.mdic.keys():
            self.mdic['TMT'] = self.mdic['TMT6plex']
            self.revmdic[self.mdic['TMT']] = 'TMT'
        elif 'TMT' in self.mdic.keys():
            self.mdic['TMT6plex'] = self.mdic['TMT']
            self.revmdic[self.mdic['TMT6plex']] = self.mdic['TMT6plex']
    
    def parseIonDictionary(self, path):
        self.ion2index = dict()
        with open(path, 'r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            for row in reader:
                self.ion2index[row[0]] = len(self.ion2index)
        self.index2ion = {b:a for a,b in self.ion2index.items()}
        self.dicsz = len(self.ion2index)

