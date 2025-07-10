import pyopenms as oms
import pyopenms.Constants
import re
import csv
import numpy as np
import torch
from annotation import annotation
import math
import statistics


def createPeptide(seq, z, mod_string):
     # create peptide sequence
    pep = oms.AASequence.fromString(seq)
    # add mods
    mods = mod_string.split("(")
    if len(mods) > 0:
        for mod in mods[1:]:
            index, residue, mod = mod.strip(")").split(",")
            pep.setModification(int(index), mod)
    return pep

def getEmpricalFormula(seq, z, mods, annot_string):
    # parse annotation
    annot = annotation.from_entry(annot_string, z)
    pep = createPeptide(seq, z, mods)
    return annot.getEmpiricalFormula(pep)

def calcIsotopeDistribution(seq, z, mods, annot_string, iso2efficiency):
    # parse annotation
    annot = annotation.from_entry(annot_string, z)
    pep = createPeptide(seq, z, mods)
    return annot.getTheoreticalIsotopeDistribution(pep, iso2efficiency)

def calcMZ(seq, z, mods, annot_string):
    # parse annotation
    annot = annotation.from_entry(annot_string, z)
    pep = createPeptide(seq, z, mods)
    return annot.getMZ(pep)

class Labeler:
    def __init__(self, D):
        self.D = D
        
    def IncompleteLabels(self, txt_fn):
        """
        Allowed modifications (230807):
        - Acetyl, Carbamidomethyl, Gln->pyro-Glu, Glu->pyro-Glu, Oxidation, 
          Phospho, Pyro-carbamidomethyl
        """
        labels_ = [a.split() for a in open(txt_fn).read().split("\n")]
        labels = []
        for data in labels_:
            seq = data[0]
            charge = int(data[1])
            mods = data[2]
            nce = int(data[3])
            
            mz = self.getPeptideWithMod(seq, mods, charge).getMonoWeight(oms.Residue.ResidueType.Full, charge) / charge
            
            label = '%s/%d_%s_NCE%.2f'%(seq,charge,mods,nce)
            labels.append(label)
        
        return labels
    
    def CompleteLabels(self, txt_fn):
        return open(txt_fn).read().split("\n")
    
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

    def apply_mask(self, targ, pred, mask, LOD, doFullMask=True):
        LOD = LOD.unsqueeze(1).expand_as(targ)
        # mask below limit of detection
        pred = torch.where(torch.logical_and(targ==0, pred<=LOD), 0.0, pred)  
        if doFullMask:
            pred = torch.where(torch.logical_and(targ==0, pred>LOD), pred-LOD, pred)

        
        # mask 1 - outside of scan range. Can have any intensity without penalty
        pred = torch.where(mask==1, 0.0, pred)
        targ = torch.where(mask==1, 0.0, targ)
        
        # mask 2-5 - bad isotope dist, below purity, high m/z error, ambiguous annotation. Can have any intensity up to the target
        if doFullMask:
            pred = torch.where(torch.logical_and(mask>1, pred < targ), 0.0, pred)
            pred = torch.where(torch.logical_and(mask>1, pred > targ), pred-targ, pred)
            targ = torch.where(mask>1, 0.0, targ)
            
            #pred = torch.where(torch.logical_and(torch.logical_and(mask>1, mask !=3), pred < targ), 0.0, pred)
            #pred = torch.where(torch.logical_and(torch.logical_and(mask>1, mask !=3), pred > targ), pred-targ, pred)
            #targ = torch.where(torch.logical_and(mask>1, mask !=3), 0.0, targ)
            
        return targ, pred
    
    
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
    
    def apply_group_constraints(self, pred, index2new_indices, new_size):
        pred = torch.nn.functional.pad(pred, (0, new_size-pred.size(dim=1)), "constant", 0)

        for index in index2new_indices:
            for new_index in index2new_indices[index]:
                pred[:, new_index] += pred[:, index]
        
        return pred
        
        #new_pred = torch.zeros((pred.size(dim=0), new_size), dtype=torch.float32)
        #new_pred[:, 0:pred.size(dim=1)] = pred
        
        #for index in index2new_indices:
        #    for new_index in index2new_indices[index]:
        #        new_pred[:, new_index] += new_pred[:, index]
        
        #return new_pred
    
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
    
    def root_intensity(self, ints, root=2):
        """
        Take the root of an intensity vector

        Parameters
        ----------
        ints : intensity vector
        root : root value

        Returns
        -------
        ints : return transformed intensity vector

        """
        if root==2:
            ints[ints>0] = torch.sqrt(ints[ints>0]) # faster than **(1/2)
        else:
            ints[ints>0] = ints[ints>0]**(1/root)
        return ints
    
    def norm_sum_one(self, ints):
        ints /= np.sum(ints)
        return ints
    
    def norm_base_peak(self, ints):
        ints /= np.max(ints)
        return ints

    def add_ce(self, label, ceadd=0, typ='nce'):
        """
        Add to collision energy in label.

        Parameters
        ----------
        label : Input label.
        ceadd : Float value to add to existing collision energy.
        typ : Either add to "ev" or "nce"

        Returns
        -------
        Label with added collision energy

        """
        hold = label.split('_')
        if len(hold)<4: hold += ['NCE0'] #TODO Non-standard label
        if typ=='ev': hold[-2] = '%.1feV'%(float(hold[-2][:-2])+ceadd)
        elif typ=='nce': hold[-1] = 'NCE%.2f'%(float(hold[-1][3:])+ceadd)
        return "_".join(hold)

    def inp_spec_msp(self, 
                     fstart, 
                     fp,
                     mint=1e-10,
                     sortmz=False
                     ):
        """
        Input spectrum from MSP file. Works on 1 spectrum at a time.
        
        :param fstart: File starting positions for "Name:..." labels in msp.
        :param fp: file pointer to msp file
        :param mint: minimum intensity for including peaks
        :param sortmz: sort the spectrum by mz, ascending. Necessary depending
                       on 
        
        :output label: spectrum label
        :output #2: tuple of (masses, intensities, ions)
        """
        fp.seek(fstart)
        label = fp.readline().split()[1]
        
        # Proceed to the peak list
        for _ in range(5):
            pos = fp.tell()
            line = fp.readline()
            if line[:9]=='Num peaks':
                fp.seek(pos)
                break
        npks = int(fp.readline().split()[2])
        masses = np.zeros((npks,));Abs = np.zeros((npks,));ions=[]
        # count=1
        for m in range(npks):
            line = fp.readline()
            # print(npks, count, line);count+=1
            spl = '\t' if '\t' in line else ' '
            split_line = line.split(spl)
            if len(split_line)==2:
                split_line += ['"?"']
            [mass,ab,ion] = split_line
            masses[m] = float(mass)
            Abs[m] = float(ab)
            ions.append(ion.strip()[1:-1].split(',')[0])
        Abs /= np.max(Abs)
        sort = Abs>mint
        return label, (masses[sort],Abs[sort],np.array(ions)[sort])
    
    def FPs(self, filename, criteria, return_labels=True):
        """
        Get file positions of spectrum labels in msp file
        
        :param filename: filepath+name of msp file
        :param criteria: string of python comparison statements to be 
                         evaluated. Use spectrum attributes "seq", "charge",
                         "ev", "nce", or "mods".
        
        :return poss: numpy array of file positions for labels meeting criteria
        """
        with open(filename,'r') as f:
            _ = f.read()
            end = f.tell()
            
            poss = []
            labs = []
            f.seek(0)
            pos = 0
            while pos<end:
                pos = f.tell()
                line = f.readline()
                # Prevent pos from blowing up (over end)
                if line=='\n':
                    pos = f.tell()
                    line = f.readline()
                if (line[:5]=='Name:') | (line[:5]=='NAME:'):
                    label = line.split()[-1].strip()
                    # if raw scans for rescoring, I expect the label to be e.g.
                    # Name: Scan=0000
                    # else then parse the label if criteria is not None
                    if len(label.split('_'))==1:
                        poss.append(pos)
                        labs.append(label)
                    else:
                        if criteria==None:
                            poss.append(pos)
                            labs.append(label)
                        else:
                            [seq,other] = line.split()[1].split('/')
                            otherspl = other.split('_') #TODO Non-standard label
                            if len(otherspl)==1: otherspl+=['0', '0eV', 'NCE0']
                            # if len(otherspl)<4: otherspl+=['NCE0'] #TODO Non-standard label
                            [charge,mods,nce] = otherspl #TODO Non-standard label
                            charge = int(charge)
                            nce = float(nce[3:])
                            if eval(criteria):
                                poss.append(pos)
                                labs.append(label)
                if line[:3]=='Num':
                    nmpks = int(line.split()[-1])
                    for _ in range(nmpks): _ = f.readline()
        if return_labels: return np.array(poss), np.array(labs)
        else: return np.array(poss)
    
    def FPs_from_labels(self, query_labels, msp_filename):
        """
        Search for matching labels in msp file, return the file position

        Parameters
        ----------
        query_labels : list of spectrum labels to search for in the msp file.
        msp_filename : Filepath+name of msp file

        Returns
        -------
        out : list of filepositions for the query labels

        """
        with open(msp_filename, 'r') as f:
            _ = f.read()
            end = f.tell()
            
            out = {label:-1 for label in query_labels}
            f.seek(0)
            count = 0
            pos = 0
            while pos<end:
                pos = f.tell()
                line = f.readline()
                if line=='\n':
                    pos = f.tell()
                    line = f.readline()
                if line[:5].upper()=='NAME:':
                    if line.strip().split()[1] in out.keys():
                        out[line.strip().split()[1]] = pos
                        count+=1
                if count==len(query_labels):
                    pos=end
        return np.array(list(out.values()))
    
    def Pos2labels(self, fname, fposs=None):
        """
        Get spectrum labels from msp file
        
        Parameters
        ----------
        fname : msp filepath+name to get labels from.
        fposs : file positions of labels. If None then use FPs() to get them.

        Returns
        -------
        labels : list of spectrum labels

        """
        labels = []
        with open(fname, 'r') as f:
            for I, pos in enumerate(fposs):
                f.seek(pos)
                label = f.readline().split()[1]
                labels.append(label)
        return labels
    
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
                pep = createPeptide(seq, 0, mods)
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