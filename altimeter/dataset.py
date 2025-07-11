from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import numpy as np
import torch as torch
import re as re
import os
import pyopenms as oms
from annotation import annotation
import statistics
from utils import getPPM

class AltimeterDataModule(LightningDataModule):
    def __init__(self, config, D):
        super().__init__()
        self.config = config
        self.D = D
        
    def getAltimeterDataset(self, dataset):
        pos_path = os.path.join(self.config['base_path'], self.config['position_path'], "fpos" + dataset + ".txt")
        data_path = os.path.join(self.config['base_path'], self.config['dataset_path'], dataset + ".txt")
        lab_path = os.path.join(self.config['base_path'], self.config['label_path'], dataset + "_labels.txt")
        return AltimeterDataset(pos_path, lab_path, data_path, self.D, num_workers=self.config['num_workers'])

    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train = self.getAltimeterDataset("train")
            self.dataset_val = self.getAltimeterDataset("val")
        elif stage == "test":
            self.dataset_test = self.getAltimeterDataset("test")
        elif stage == "predict":
            self.dataset_test = self.getAltimeterDataset("val")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=self.config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=self.config['num_workers'])
    
    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=1)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

class AltimeterDataset(Dataset):
    def __init__(self, pos_path, label_path, data_path, dobj, num_workers=1, transform=None):
        
        self.positions = np.loadtxt(pos_path).astype(int)
        self.labels = np.array([line.strip() for line in open(label_path,'r')])
        self.worked_id2fdata = dict()
        for worker_id in range(num_workers):
             self.worked_id2fdata[worker_id] = open(data_path, "r")
        
        self.D = dobj
        self.channels = dobj.seq_channels
        self.transform = transform
        
        

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Handle single-worker case
            worker_id = 0
        else:
            worker_id = worker_info.id
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples, info = self.input_from_str(self.labels[idx])
        seq, mod, charge, nce, min_mz, max_mz, LOD, weight = info
        targ, _, mask = self.target(self.worked_id2fdata[worker_id], self.positions[idx], return_mz=False)
        sample = {'samples': samples, 'targ': targ, 'mask': mask, 'seq': seq, 'mod': mod, 'charge': charge, 'nce': nce, 'min_mz': min_mz, 'max_mz': max_mz, 'LOD': LOD, 'weight': weight}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_target_plot(self, idx):
        samples, info = self.input_from_str(self.labels[idx])
        seq, mod, charge, nce, min_mz, max_mz, LOD, weight = info
        targ, moverz, annotated, mask = self.target_plot(self.worked_id2fdata[0], self.positions[idx])
        
        return [samples, targ, mask, seq, mod, charge, nce, min_mz, max_mz, LOD, weight, moverz, annotated]
    
    def input_from_str(self, strings):
        [seq,other] = strings.split('/')
        osplit = other.split("_")
       
        [charge, mod, nce, scan_range, LOD, weight] = osplit
        charge = int(charge)
        nce = float(nce[3:])
        min_mz = float(scan_range.split("-")[0])
        max_mz = float(scan_range.split("-")[1])
        LOD = float(LOD)
        weight = float(weight)
            
        info = (seq, mod, charge, nce, min_mz, max_mz, LOD, weight)
        out = self.inptsr(info)
        return out, info
    
    def target(self, fp, fstart, mint=0, return_mz=False, return_annotated=False):
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
            (1, self.D.dicsz), mint, dtype=torch.float32
        )
        moverz = (torch.zeros((1, self.D.dicsz), dtype=torch.float32) if return_mz else 0)
        masks = torch.full((1, self.D.dicsz), mint, dtype=torch.float32)
        if return_annotated:
            annotated = torch.zeros((1, self.D.dicsz), dtype=torch.float32)
        
        # Fill the output
        fp.seek(fstart)
        nmpks = int(fp.readline().split()[1].split("|")[-1])
        for pk in range(nmpks):
            line = fp.readline()
            [d,I,mz,intensity,mask] = line.split()
            I = int(I)
            if I == -1: continue
            target[0,I] = float(intensity)
            masks[0,I] = int(mask)
            if return_mz: moverz[0,I] = float(mz)

        return target, moverz, masks
    
    def target_plot(self, fp, fstart, mint=0):
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
        (seq, mod, charge, nce, _, _, _, _) = info
        output = torch.zeros((self.channels, self.D.seq_len), dtype=torch.float32)
        
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
        
        out = [output, torch.tensor(float(charge)), torch.tensor(float(nce))]
        return out
    
    def ConvertToPredictedSpectrum(self, pred, seq, mod, charge):
        
        pred_ions = np.array([self.D.index2ion[ind] for ind in range(len(self.D.index2ion))])
        
        # zero out impossible ions
        filt = self.filter_fake(seq, mod, charge, pred_ions)

        valid_pred = pred[filt]
        valid_pred_ions = pred_ions[filt]
        
        # create new targ and mass np arrays with expected isotope size
        size_with_isotopes = valid_pred.size
        pred_full = np.zeros(size_with_isotopes)
        mz_full = np.zeros(size_with_isotopes)
        ions_full = np.empty(size_with_isotopes, dtype=object)

        # populate with predicted mono isotopes
        for i, ion_total_intensity in enumerate(valid_pred):
            if "^" in valid_pred_ions[i]:
                ion_charge = int(valid_pred_ions[i].split("^")[-1])
            else:
                ion_charge = 1
            mono_mz = calcMZ(seq, ion_charge, mod, valid_pred_ions[i])
 
            pred_full[i] = ion_total_intensity
            mz_full[i] = mono_mz
            ions_full[i] = valid_pred_ions[i]
        
        # return intensities and m/z's
        pred_full /= pred_full.max()
        
        return pred_full, mz_full, ions_full.astype('U')
    
    
    def filter_fake(self, seq, mods, charge, ions):        
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
                        
            if "m" == ion[0]:
                [start,ext] = [
                    int(j) for j in 
                    ion[1:].split("^")[0].split('+')[0].split('-')[0].split(':')
                ]
                # Do not write if the internal extends beyond length of peptide-2
                if (start+ext)>=(len(seq)-2): a = False
            if (
                (ion[0] in ['a','b','y']) and 
                (int(ion[1:].split('-')[0].split('+')[0].split('^')[0])>(len(seq)-1))
                ):
                # Do not write if a/b/y is longer than length-1 of peptide
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
    
    

def calcMZ(seq, z, mods, annot_string):
    # parse annotation
    annot = annotation.from_entry(annot_string, z)
    pep = createPeptide(seq, mods)
    return annot.getMZ(pep)

def createPeptide(seq, mod_string):
     # create peptide sequence
    pep = oms.AASequence.fromString(seq)
    # add mods
    mods = mod_string.split("(")
    if len(mods) > 0:
        for mod in mods[1:]:
            index, residue, mod = mod.strip(")").split(",")
            pep.setModification(int(index), mod)
    return pep

def filter_by_scan_range(mz, ints, min_mz, max_mz, annotated=None):
    ints[mz < min_mz] = 0
    ints[mz > max_mz] = 0
    if annotated is not None:
        annotated = annotated[ints != 0]
    return ints[ints != 0], mz[ints != 0], annotated

def match(int_targ, mz_targ, int_pred, mz_pred):
        ppm_tol=20
        # collapse predicted on itself
        mz_aligned = []
        int_pred_aligned = []
        current_sum = 0.0
        current_group = []
        i = 0
        while i < (mz_pred.size-1):
            if abs(getPPM(mz_pred[i], mz_pred[i+1])) > ppm_tol:
                current_group.append(mz_pred[i])
                mz_aligned.append(statistics.fmean(current_group))
                int_pred_aligned.append(current_sum + int_pred[i])
                current_sum = 0.0
                current_group.clear()
            else:
                current_group.append(mz_pred[i])
                current_sum += int_pred[i]
            i+=1
        # last element corner case
        current_group.append(mz_pred[i])
        mz_aligned.append(statistics.fmean(current_group))
        int_pred_aligned.append(current_sum + int_pred[i])
        
        int_pred_aligned = np.array(int_pred_aligned, dtype=np.float32)
        mz_aligned = np.array(mz_aligned, dtype=np.float32)
        
        # align targ onto predicted
        int_targ_aligned = np.zeros_like(int_pred_aligned)
        targ_index = 0
        for i in range(int_targ_aligned.size-1):
            ppm_cur = abs(getPPM(mz_aligned[i], mz_targ[targ_index]))
            ppm_next = abs(getPPM(mz_aligned[i+1], mz_targ[targ_index]))
            
            while ppm_cur < ppm_next and targ_index < int_targ.size:
                int_targ_aligned[i] += int_targ[targ_index]
                targ_index += 1
                if targ_index >= int_targ.size: 
                    break
                ppm_cur = abs(getPPM(mz_aligned[i], mz_targ[targ_index]))
                ppm_next = abs(getPPM(mz_aligned[i+1], mz_targ[targ_index]))
            
            if targ_index >= int_targ.size: 
                break
                
        while targ_index < int_targ.size:
            int_targ_aligned[int_targ_aligned.size-1] += int_targ[targ_index]
            targ_index += 1
                        
        return int_targ_aligned, int_pred_aligned, mz_aligned
    
def norm_base_peak(ints):
    ints /= np.max(ints)
    return ints