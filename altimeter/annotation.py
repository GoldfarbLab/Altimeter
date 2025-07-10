import pyopenms as oms
import numpy as np
import re as re

IMMONIUM_ION_FORMULA = {
    'IA' : "C2H5N",
    'IC' : "C2H5NS",
    'IC(Carbamidomethyl)' : "C4H8N2OS",
    'ID' : "C3H5NO2",
    'IE' : "C4H7NO2",
    'IF' : "C8H9N",
    'IG' : "CH3N",
    'IH' : "C5H7N3",
    'II' : "C5H11N",
    'IK' : "C5H12N2",
    'IL' : "C5H11N",
    'IM' : "C4H9NS",
    'IM(Oxidation)' : "C4NOSH9",
    'IN' : "C3H6N2O",
    'IP' : "C4H7N",
    'IQ' : "C4H8N2O",
    'IR' : "C5H12N4",
    'IS' : "C2H5NO",
    'IT' : "C3H7NO",
    'IV' : "C4H9N",
    'IW' : "C10H10N2",
    'IY' : "C8H9NO"
}

class annotation:
    
    def __init__(self, name, NL, NG, frag_z, frag_iso, error):
        self.name = name
        self.NL = NL
        self.NG = NG
        self.z = frag_z
        self.isotope = frag_iso
        self.error = error
        self.length = None
        if self.name[0] in "abcxyz":
            self.length = int(self.name[1:])
        elif self.name[0] == "m":
            [start, end] = self.name[1:].split(":")
            self.length = 1 + int(end) - int(start)
        else:
            self.length = 1

    @classmethod  
    def from_entry(cls, entry, prec_z):
        if entry == "?": 
            return cls("?", None, None, None, None, None)
        
        if "/" in entry:
            error = float(entry.split("/")[-1][:-3])
            entry = entry.split("/")[0]
        else:
            error = 0.0#None
        
        # get fragment isotope
        isotope = entry.split("+")[-1]
        if isotope[-1] == "i":
            if isotope[:-1] == "":
                isotope = 1
            else:
                isotope = int(isotope[:-1])
            entry = "+".join(entry.split("+")[0:-1])
        else:
            isotope = 0
    
        # get fragment charge
        if "^" not in entry:
            z = 1
        else:
            z = int(entry.split("^")[1][0])
        entry = entry.split("^")[0]
        
        # get neutral gain
        if "+" in entry and not "-" in entry.split("+")[-1]:
            NG = entry.split("+")[1:]
            entry = entry.split("+")[0]
        else:
            NG = None
        
        # get neutral loss
        if "-" in entry:
            NL = entry.split("-")[1:]
            entry = entry.split("-")[0]
        else:
            NL = None
        
        # check if neutral gain was out of order
        if "+" in entry:
            NG = entry.split("+")[1:]
            entry = entry.split("+")[0]

        # get remaining name
        name = entry
        return annotation(name, NL, NG, z, isotope, error)
        
    def toString(self):
        if self.name == "?": return "?"
        out = self.getName()
        if self.isotope > 0:
            out += "+" 
            if self.isotope > 1:
                out += str(self.isotope)
            out += "i"
        if self.error is not None:
            out += "/" + "{:.1f}".format(self.error) + "ppm"
        return out
    
    def getName(self):
        if self.name == "?": return "?"
        out = self.name
        out += self.getNLString()
        if self.z > 1:
            out += "^"+str(self.z)
        return out
    
    def getIsoName(self):
        if self.name == "?": return "?"
        out = self.getName()
        if self.isotope > 0:
            out += "+" 
            if self.isotope > 1:
                out += str(self.isotope)
            out += "i"
        return out
    
    def getMonoMass(self, pep):
        frag_formula = self.getEmpiricalFormula(pep)
        return frag_formula.getMonoWeight()
    
    def getMZ(self, pep):
        frag_formula = self.getEmpiricalFormula(pep)
        
        # return mz
        mz = (frag_formula.getMonoWeight() / self.z) + ((self.isotope * oms.Constants.C13C12_MASSDIFF_U) / self.z)
    
        return mz
    
    def isValidAnnot(self, pep):
        return "-" not in self.getEmpiricalFormula(pep).toString()
        
    def hasNL(self):
        return (self.NL and len(self.NL) > 0) or (self.NG and len(self.NG) > 0)
    
    def getType(self):
        if self.name[0] == "p": return "p"
        if self.name[0] == "b": return "b"
        if self.name[0] == "y": return "y"
        if self.name[0] == "m": return "m"
        if self.name[0] == "I": return "Imm"
        
    def getNLString(self):
        out = ""
        if self.NL:
            out += "-" + "-".join(self.NL)
        if self.NG:
            out += "+" + "+".join(self.NG)
        return out
    
    def getEmpiricalFormula(self, pep):
        # get elemental composition
        if self.name[0] == "p":
            frag_formula = pep.getFormula(oms.Residue.ResidueType.Full, self.z)
        elif self.name[0] in "abcxyz":
            length = int(self.name[1:])
            if self.name[0] == "a":
                frag_formula = pep.getPrefix(length).getFormula(oms.Residue.ResidueType.AIon, self.z)
            elif self.name[0] == "b":
                frag_formula = pep.getPrefix(length).getFormula(oms.Residue.ResidueType.BIon, self.z)
            elif self.name[0] == "c":
                frag_formula = pep.getPrefix(length).getFormula(oms.Residue.ResidueType.CIon, self.z)
            elif self.name[0] == "x":
                frag_formula = pep.getSuffix(length).getFormula(oms.Residue.ResidueType.XIon, self.z)
            elif self.name[0] == "y":
                frag_formula = pep.getSuffix(length).getFormula(oms.Residue.ResidueType.YIon, self.z)
            elif self.name[0] == "z":
                frag_formula = pep.getSuffix(length).getFormula(oms.Residue.ResidueType.ZIon, self.z)
        elif self.name[0] == "m":
            frag_start = int(self.name[1:].split(":")[0])
            frag_formula = pep.getSubsequence(frag_start, self.length).getFormula(oms.Residue.ResidueType.Internal, self.z)
    
        # deal with immonium
        else:
            frag_formula = oms.EmpiricalFormula(IMMONIUM_ION_FORMULA[self.name])
            frag_formula.setCharge(self.z)
        
        # deal with NLs / NGs
        if self.NL:
            for nl in self.NL:
                if nl[0].isdigit():
                    nl_count = int(re.search("^\d*", nl).group(0))
                    for j in range(nl_count):
                        frag_formula -= oms.EmpiricalFormula(nl[len(str(nl_count)):])
                else:
                    frag_formula -= oms.EmpiricalFormula(nl)

        if self.NG:
            for ng in self.NG:
                if ng[0].isdigit():
                    ng_count = int(re.search("^\d*", ng).group(0))
                    for j in range(ng_count):
                        frag_formula += oms.EmpiricalFormula(ng[len(str(ng_count)):])
                else:
                    frag_formula += oms.EmpiricalFormula(ng)
        
        return frag_formula
    
    def getTheoreticalIsotopeDistribution(self, pep, iso2efficiency):
        
        pep_formula = pep.getFormula()
        frag_formula = self.getEmpiricalFormula(pep)
        
        pep_iso_gen = oms.CoarseIsotopePatternGenerator(1+max(iso2efficiency.keys()))
        pep_iso_dist = pep_formula.getIsotopeDistribution(pep_iso_gen)
        
        weighted_frag_dist = np.zeros(1+max(iso2efficiency.keys()))
        
        if pep_formula.toString() == frag_formula.toString():
            for iso_index in range(len(pep_iso_dist.getContainer())):
                if iso_index in iso2efficiency:
                    weighted_frag_dist[iso_index] += pep_iso_dist.getContainer()[iso_index].getIntensity() * iso2efficiency[iso_index]
        
        else:
            for iso in iso2efficiency:
                igen = oms.CoarseIsotopePatternGenerator(1+iso)
                frag_iso_dist = frag_formula.getConditionalFragmentIsotopeDist(pep_formula, {iso}, igen)
                
                for iso_index in range(len(frag_iso_dist.getContainer())):
                    weighted_frag_dist[iso_index] += pep_iso_dist.getContainer()[iso].getIntensity() * iso2efficiency[iso] * frag_iso_dist.getContainer()[iso_index].getIntensity()

        return weighted_frag_dist
    
    def getMostAbundantIsotope(self, pep, iso2efficiency):
        return np.argmax(self.getTheoreticalIsotopeDistribution(pep, iso2efficiency))
    
    def getMostAbundantIsotopes(self, pep, iso2efficiency, threshold=0.8):
        isotope_dist = self.getTheoreticalIsotopeDistribution(pep, iso2efficiency)
        return [i for i, prob in enumerate(isotope_dist) if prob/max(isotope_dist) >= threshold]
            
             


class annotation_list:
    def __init__(self, annotations):
        self.entries = annotations
        
    def annotationString(self):
        if len(self.entries) == 0:
            return "?"
        elif len(self.entries) == 1 and self.entries[0].toString() == "?":
            return "?"
        return ",".join([annot.toString() for annot in self.entries])
    
    def annotationName(self):
        if len(self.entries) == 0:
            return "?"
        elif len(self.entries) == 1 and self.entries[0].toString() == "?":
            return "?"
        return ",".join([annot.getName() for annot in self.entries])
    
    def getBestEntry(self):
        if len(self.entries) == 0:
            return None
        sorted_entries = sorted(self.entries, key=lambda x: abs(x.error))
        current_best_error = sorted_entries[0].error
        current_best = sorted_entries[0]
        for i in range(1, len(sorted_entries)):
            if sorted_entries[i].error == current_best_error:
                if sorted_entries[i].name == "y1":
                    current_best = sorted_entries[i]
        return current_best