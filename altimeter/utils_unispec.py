import pyopenms as oms
import pyopenms.Constants
import re
import csv
import numpy as np
import torch
from annotation import annotation
from dataset import createPeptide, calcMZ
import math
import statistics


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
        




class LoadObj:
    def __init__(self, dobj, embed=False):
        self.D = dobj
        self.embed = embed
        self.channels = dobj.seq_channels if embed else dobj.channels
    
    def str2dat(self, string):
        """
        Turn a label string into its constituent 

        Parameters
        ----------
        string : label string in form {seq}/{charge}_{mods}_{ev}eV_NCE{nce}

        Returns
        -------
        Tuple of seq,mods,charge,ev,nce

        """
        seq,other = string.split('/')
        [charge,mods,nce] = other.split('_')
        # Mstart = mods.find('(') if mods!='0' else 1
        # modnum = int(mods[0:Mstart])
        # if modnum>0:
        #     modlst = [re.sub('[()]','',m).split(',') 
        #               for m in mods[Mstart:].split(')(')]
        #     modlst = [(int(m[0]),m[-1]) for m in modlst]
        # else: modlst = []
        return (seq,mods,int(charge),float(nce[3:]))
    
    def inptsr(self, info):
        """
        Create input(s) for 1 peptide

        Parameters
        ----------
        info : tuple of (seq,mod,charge,nce)

        Returns
        -------
        out : List of a) tensor to model, b) charge float and/or c) ce float.
              Only outputs a) if not self.embed

        """
        (seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) = info
        output = torch.zeros((self.channels, self.D.seq_len), dtype=torch.long)
        
        # Sequence
        assert len(seq) <= self.D.seq_len, "Exceeded maximum peptide length."
        output[:len(self.D.dic),:len(seq)] = torch.nn.functional.one_hot(
            torch.tensor([self.D.dic[o] for o in seq], dtype=torch.long),
            len(self.D.dic)
        ).T
        output[len(self.D.dic)-1, len(seq):] = 1.
        # PTMs
        Mstart = mod.find('(') if mod!='0' else 1
        modamt = int(mod[0:Mstart])
        output[len(self.D.dic)] = 1.
        if modamt>0:
            hold = [re.sub('[()]', '', n) for n in mod[Mstart:].split(")(")]
            for n in hold:
                [pos,aa,modtyp] = n.split(',')
                output[self.D.mdic[modtyp], int(pos)] = 1.
                output[len(self.D.dic), int(pos)] = 0.
        
        if self.embed:
            out = [output, float(charge), float(nce)]
        if not self.embed:
            output[self.D.seq_channels+int(charge)-1] = 1. # charge
            output[-1, :] = float(nce)/100. # ce
            out = [output]
        return out

    
    
    def input_from_file(self, fstarts, fn):
        """
        Create batch of model inputs from array of file starting positions and
        the filename.
        - If self.embed=True, then this function outputs charge and energy as 
        batch-size length arrays containing the their respective values for 
        each input. Otherwise the first output is just a single embedding tensor.
        
        :param fstarts: array of file postions for spectral labels to be loaded.
        :param fn: filename to be opened
        
        :return out: List of inputs to the model. Length is 3 if
                     self.embed=True, else length is 1.
        :return info: List of tuples of peptide data. Each tuple is ordered as
                      (seq,mod,charge,nce).
        """
        if type(fstarts)==int: fstarts = [fstarts]

        bs = len(fstarts)
        outseq = torch.zeros((bs, self.channels, self.D.seq_len),
                             dtype=torch.float32)
        if self.embed:
            outch = torch.zeros((bs,), dtype=torch.float32)
            outce = torch.zeros((bs,), dtype=torch.float32)


        info = []
        with open(fn,'r') as fp:
            for m in range(len(fstarts)):
                fp.seek(fstarts[m])
                line = fp.readline()
                [seq, mod, charge, nce, scan_range, LOD, iso_efficiency, nmpks] = line.split()[1].split("|")
                charge = int(charge)
                print("pre", nce)
                nce = float(nce[:-3])
                print("post", nce)
                min_mz = float(scan_range.split("-")[0])
                max_mz = float(scan_range.split("-")[1])
                LOD = float(LOD)
                iso2efficiency = dict()
                for iso_entry in iso_efficiency.split(","):
                    isotope = int(iso_entry.split(")")[-1])
                    efficiency = float(iso_entry.split(")")[0][1:])
                    iso2efficiency[isotope] = efficiency
                info.append((seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency)) # dummy 0 for nce
                out = self.inptsr(info[-1])
                outseq[m] = out[0]
                if self.embed:
                    outch[m] = out[1]
                    outce[m] = out[2]
        out = [outseq, outch, outce] if self.embed else [outseq]
        return out, info
    
    def input_from_str(self, strings):
        """
        Create batch of model inputs from list of string input labels. 
        - If self.embed=True, then this function outputs charge and energy as 
        batch-size length arrays containing the their respective values for 
        each input. Otherwise the first output is just a single embedding tensor.
        
        :param strings: List of input labels. All input labels must have the
                        form {seq}/{charge}_{mods}_NCE{nce}
        
        :return out: List of inputs to the model. Length is 3 if
                     self.embed=True, else length is 1.
        :return info: List of tuples of peptide data. Each tuple is ordered as
                      (seq,mod,charge,nce).
        """
        if (type(strings)!=list)&(type(strings)!=np.ndarray): 
            strings = [strings]
        
        bs = len(strings)
        outseq = torch.zeros(
            (bs, self.channels, self.D.seq_len), dtype=torch.float32
        )
        if self.embed:
            outch = torch.zeros((bs,), dtype=torch.float32)
            outce = torch.zeros((bs,), dtype=torch.float32)

        info = []
        for m in range(len(strings)):
            [seq,other] = strings[m].split('/')
            osplit = other.split("_") #TODO Non-standard label
            #if len(osplit)==5: osplit+=['NCE0'] #TODO Non-standard label
            [charge, mod, nce, scan_range, LOD, iso_efficiency, weight] = osplit#other.split('_') #TODO Non-standard label
            charge = int(charge)
            nce = float(nce[3:])
            min_mz = float(scan_range.split("-")[0])
            max_mz = float(scan_range.split("-")[1])
            LOD = float(LOD)
            weight = float(weight)
            iso2efficiency = dict()
            for iso_entry in iso_efficiency.split(","):
                if iso_entry == "": iso2efficiency[0] = 1
                else:
                    isotope = int(iso_entry.split(")")[-1])
                    efficiency = float(iso_entry.split(")")[0][1:])
                    iso2efficiency[isotope] = efficiency
                
            info.append((seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight))
            out = self.inptsr(info[-1])
            outseq[m] = out[0]
            if self.embed:
                outch[m] = out[1]
                outce[m] = out[2]
        
       # out = [outseq, outch, outce] if self.embed else [outseq]
        out = [outseq, outch] if self.embed else [outseq] #FIXME !!! go back to previous
        return out, info
    
    def target(self, fstart, fp, mint=0, return_mz=False):
        """
        Create target, from streamlined dataset, to train model on.
        
        :param fstart: array of file positions for spectra to be predicted.
        :param fp: filepointer to streamlined dataset.
        :param mint: minimum intensity to include in target spectrum.
        :param return_mz: whether to return the corresponding m/z values for
                          fragment ions.
        
        :return target: pytorch array of intensities for all ions in output
                        output space.
        :return moverz: pytorch array of m/z values corresponding to ions
                        present in target array. All zeros if return_mz=False.
        """
        
        target = torch.full(
            (len(fstart), self.D.dicsz), mint, dtype=torch.float32
        )
        moverz = (torch.zeros((len(fstart), self.D.dicsz), dtype=torch.float32) if return_mz else 0)
        masks = torch.full((len(fstart), self.D.dicsz), mint, dtype=torch.float32)
        
        # Fill the output
        for i,p in enumerate(fstart):
            fp.seek(p)
            nmpks = int(fp.readline().split()[1].split("|")[-1])
            for pk in range(nmpks):
                line = fp.readline()
                [d,I,mz,intensity,mask] = line.split()
                I = int(I)
                if I == -1: continue
                target[i,I] = float(intensity)
                masks[i,I] = int(mask)
                if return_mz: moverz[i,I] = float(mz)

        return target, moverz, masks
    
    def target_plot(self, fstart, fp, mint=0):
        target = []
        moverz = []
        annotated = []
        masks = []
        fp.seek(fstart)
        nmpks = int(fp.readline().split()[1].split("|")[-1])
        for pk in range(nmpks):
            [d,I,mz,intensity,mask] = fp.readline().split()
            if float(intensity) > 0:
                target.append(float(intensity))
                moverz.append(float(mz))
                masks.append(int(mask))
                annotated.append(I != "-1")
        target = np.array(target, dtype=np.float32)
        mz = np.array(moverz, dtype=np.float32) 
        annotated = np.array(annotated, dtype=np.bool_) 
        masks = np.array(masks, dtype=np.int32)
        target /= np.max(target)
        
        return target, mz, annotated, masks
    

    
    def ConvertToPredictedSpectrum(self, pred, info, doIso=True):
        (seq, mod, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) = info
        
        pred_ions = np.array([self.D.index2ion[ind] for ind in range(len(self.D.index2ion))])
        
        # zero out impossible ions
        filt = self.filter_fake(info, pred_ions)

        valid_pred = pred[filt]
        valid_pred_ions = pred_ions[filt]
        
        # create new targ and mass np arrays with expected isotope size
        num_isotopes = max(iso2efficiency)+1
        size_with_isotopes = valid_pred.size * num_isotopes
        pred_full = np.zeros(size_with_isotopes)
        mz_full = np.zeros(size_with_isotopes)
        ions_full = np.empty(size_with_isotopes, dtype=object)

        # populate with predicted mono isotopes
        for i, ion_total_intensity in enumerate(valid_pred):
            if valid_pred_ions[i][0] == "p": ion_charge = charge
            else:
                ion_charge = int(valid_pred_ions[i].split("^")[-1])
            mono_mz = calcMZ(seq, charge, mod, valid_pred_ions[i])
            # predict isotope distribution
            if doIso:
                ion_isotope_dist = calcIsotopeDistribution(seq, charge, mod, valid_pred_ions[i], iso2efficiency)
                for iso_index, iso_prob in enumerate(ion_isotope_dist):
                    if math.isnan(iso_prob): iso_prob = 0
                    pred_full[(i * num_isotopes) + iso_index] = iso_prob * ion_total_intensity
                    mz_full[(i * num_isotopes) + iso_index] = mono_mz + (iso_index * pyopenms.Constants.C13C12_MASSDIFF_U) / ion_charge
                    ions_full[(i * num_isotopes) + iso_index] = valid_pred_ions[i]
            else:
                pred_full[(i * num_isotopes)] = ion_total_intensity
                mz_full[(i * num_isotopes)] = mono_mz
                ions_full[(i * num_isotopes)] = valid_pred_ions[i]
        
        # return intensities and m/z's
        pred_full /= pred_full.max()
        
        return pred_full, mz_full, ions_full.astype('U')
        
    
    
    def filter_fake(self, pepinfo, ions):
        """
        Filter out the ions which cannot possibly occur for the peptide being
         predicted.

        Parameters
        ----------
        pepinfo : tuple of (sequence, mods, charge, ev, nce). Identical to
                   second output of str2dat().
        masses: array of predicted ion masses
        ions : array or list of predicted ion strings.

        Returns
        -------
        Return a numpy boolean array which you can use externally to select
         indices of m/z, abundance, or ion arrays

        """
        (seq, mods, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) = pepinfo
        
        #print(mods)
        # modification
        modlst = []
        Mstart = mods.find('(') if mods!='0' else 1
        modamt = int(mods[0:Mstart])
        if modamt>0:
             Mods = mods[Mstart:].split(')(') # )( always separates modifications
             for mod in Mods:
                 [pos,aa,typ] = re.sub('[()]', '', mod).split(',') # mod position, amino acid, and type
                 modlst.append([int(pos), aa, typ])
        
        filt = []
        for ion in ions:
            annot = annotation.from_entry(ion, charge)
            
            ion_charge = annot.z
            ion_type = annot.getType()
            ext = annot.length if ion_type != "p" else len(seq)
           
            a = True
            # Do not include immonium ions for amino acids missing from the sequence
            if ion_type == "Imm":
                a = False
                if annot.getName()[1] in seq:
                    a = True
                if "IC(Carbamidomethyl)" in ion:
                    if 'Carbamidomethyl' not in mods:
                        a = False
                if "IM(Oxidation)" in ion:
                    if "Oxidation" not in mods:
                        a = False
                        
            #if "m" in ion:
            #    [start,ext] = [
            #        int(j) for j in 
            #        ion[1:].split("^")[0].split('+')[0].split('-')[0].split(':')
            #    ]
                # Do not write if the internal extends beyond length of peptide-2
           #     if (start+ext)>=(len(seq)-2): a = False
            # The precursor ion must be the same charge
            #if ion[0] == 'p' and ion_charge != charge:
            #    a = False
            if (
                (ion[0] in ['a','b','y']) and 
                (int(ion[1:].split('-')[0].split('+')[0].split('^')[0])>(len(seq)-1))
                ):
                # Do not write if a/b/y is longer than length-1 of peptide
                a = False
            if ('H3PO4' in ion):
                # Do not write Phospho specific neutrals for non-phosphopeptide
                nl_count = 1
                for nl in annot.NL:
                    if 'CH4SO' in nl:
                        if nl[0].isdigit():
                            nl_count = int(re.search("^\d*", nl).group(0))
                if sum(['Phospho' == mod for mod in mods]) < nl_count:
                    a = False
            if ('CH4SO' in ion):
                nl_count = 1
                for nl in annot.NL:
                    if 'CH4SO' in nl:
                        if nl[0].isdigit():
                            nl_count = int(re.search("^\d*", nl).group(0))
                if self.getModCount(seq, ion, ext, 'Oxidation', modlst) < nl_count:
                    a = False
                    
            if ('C2H5SNO' in ion):
                nl_count = 1
                for nl in annot.NL:
                    if 'CH4SO' in nl:
                        if nl[0].isdigit():
                            nl_count = int(re.search("^\d*", nl).group(0))
                if self.getModCount(seq, ion, ext, 'Carbamidomethyl', modlst) < nl_count:
                    a = False
            # Do not include fragments with a higher charge than the precursor
            if ion_charge > charge:
                a = False
            
            if a:
                pep = createPeptide(seq, mods)
                ec = annot.getEmpiricalFormula(pep).getElementalComposition()
                
                for element in ec:
                    if ec[element] < 0:
                        a = False
                
            filt.append(a)

        return np.array(filt)
    
    def getModCount(self, seq, ion, ext, mod_target, mods):
        count = 0
        if ion[0] == 'b':
            for pos, aa, mod_type in mods:
                if mod_type == mod_target and pos < ext:
                    count+=1
        elif ion[0] == 'y':
             for pos, aa, mod_type in mods:
                if mod_type == mod_target and pos >= len(seq) - ext:
                    count+=1
        elif ion.startswith("m"):
            if ion[1].isdigit():
                [start,ext] = [
                        int(j) for j in 
                        ion[1:].split("^")[0].split('+')[0].split('-')[0].split(':')
                    ]
                for pos, aa, mod_type in mods:
                    if mod_type == mod_target and pos >= start and pos < start + ext:
                        count+=1
            else:
                count = ion.count(mod_target)
        return count
