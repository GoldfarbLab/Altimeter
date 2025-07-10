import numpy as np
from numpy.linalg import norm
import pyopenms as oms
from collections import defaultdict

def cosineSim(A,B):
    return np.dot(A,B)/max(1e-8,(norm(A)*norm(B)))

def getPPMAbs(mz_obs, mz_ref):
    return abs(getPPM(mz_obs, mz_ref))

def getPPM(mz_obs, mz_ref):
    return (mz_obs - mz_ref) / mz_ref * 1e6

def ppmToMass(ppm, mz_ref):
    return (ppm / 1e6) * mz_ref

def isoDist2np(dist):
    iso_prob = np.zeros(len(dist.getContainer()))
    for i in range(len(dist.getContainer())):
        iso_prob[i] =  dist.getContainer()[i].getIntensity()
    return iso_prob

def linearInterpolate(x, x_left, y_left, x_right, y_right):
    if x_left == -1: return y_right
    if x_right == -1: return y_left
    
    percent_right = (x - x_left) / (x_right - x_left)
    diff_y = y_right - y_left
    return (diff_y * percent_right) + y_left

def pepFromSage(mod_seq):
    mod_seq = mod_seq.replace("]-", "]")
    mod_seq = mod_seq.replace("-[", "[")
    return oms.AASequence.fromString(mod_seq)


def get_mod_seq(seq, mods):
    mods_oms = mods.replace("Unimod:4", "Carbamidomethyl").replace("Unimod:35", "Oxidation")
    mods_oms = mods.replace("UNIMOD:4", "Carbamidomethyl").replace("UNIMOD:35", "Oxidation")
    corrected_mods = ""
    if len(mods_oms) > 0:
        index2mod = defaultdict(list)
        for mod in mods_oms.split("(")[1:]:
            mod = mod.strip(")")
            index, aa, ptm = mod.split(",")
            index = int(index)-1
            index2mod[index].append(ptm)
            corrected_mods += "(" + str(index) + "," + aa + "," + ptm + ")"
        mod_seq = ""
        for i, aa in enumerate(seq):
            if i in index2mod:
                for mod in index2mod[i]:
                    if mod == "Acetyl":
                        mod_seq += "(" + mod + ")"
            mod_seq += aa
            if i in index2mod:
                for mod in index2mod[i]:
                    if mod != "Acetyl":
                        mod_seq += "(" + mod + ")"
        peptide = oms.AASequence.fromString(mod_seq)
        mods_oms = str(len(mods_oms.split("("))-1) + mods_oms
        corrected_mods = str(len(mods_oms.split("("))-1) + corrected_mods
    else:
        peptide = oms.AASequence.fromString(seq)
        mods_oms = '0'
                
    return peptide, mods_oms