import csv

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
    
    def parseIonDictionary(self, path):
        self.ion2index = dict()
        with open(path, 'r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            for row in reader:
                self.ion2index[row[0]] = len(self.ion2index)
        self.index2ion = {b:a for a,b in self.ion2index.items()}
        self.dicsz = len(self.ion2index)

