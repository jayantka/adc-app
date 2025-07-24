#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:38:24 2025

@author: jayant
"""

import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, DataStructs
import py3Dmol
import pandas as pd
import os
from io import BytesIO
from PIL import Image

# Prepare image folder
os.makedirs("adc_images", exist_ok=True)

# Public linkers
public_linkers = {
    "Linker A": "CCOC(=O)C",
    "Linker B": "CCN(CC)CC",
    "Linker C": "CC(=O)NCC",
    "Linker D": "COC(=O)CCN",
    "Linker E": "CC(C)COC(=O)",
    "Linker F": "CCOC(=O)N(C)C",
    "Linker G": "CN(C)C(=O)CC",
    "Linker H": "CCC(=O)OC",
    "Linker I": "CNC(=O)OC",
    "Linker J": "CCN(C)C(=O)"
}
public_fps = {name: Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for name, smi in public_linkers.items()}

def find_similar_linker(user_linker):
    user_mol = Chem.MolFromSmiles(user_linker)
    if not user_mol:
        return "Invalid SMILES", 0.0
    user_fp = Chem.RDKFingerprint(user_mol)
    best_name = None
    best_sim = -1.0
    for name, fp in public_fps.items():
        sim = DataStructs.FingerprintSimilarity(user_fp, fp)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name, round(best_sim, 3)

def contains_ester(mol):
    return mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)OC'))

def contains_amide(mol):
    return mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N'))

def predict_stability(mol):
    if contains_amide(mol):
        return "Stable"
    elif contains_ester(mol):
        return "Unstable"
    else:
        return "Moderate"

def predict_release_profile(logp, molwt):
    if logp > 4.5:
        return "Slow-release"
    elif molwt < 400:
        return "Fast-release"
    else:
        return "pH-sensitive"

def mock_simulation(qed, logp, molwt):
    cost = 1000 + 0.5 * molwt
    timeline = 30 + 10 * (logp - qed)
    risk = "High" if qed < 0.4 or logp > 5 else "Low"
    return round(cost, 1), round(timeline, 1), risk

def mol_to_image(mol):
    img = Draw.MolToImage(mol, size=(300, 300))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def show_3d_molecule(mol):
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    return viewer._make_html()

# ------------------- Streamlit UI -------------------
st.title("ADC Molecule Generator with Linker Repurposing")

# Inputs
warhead = st.text_input("Warhead SMILES", value="CC(C1=CC=CC=C1)NC2=NC=NC=C2")
payload = st.text_input("Payload SMILES", value="CN(C)C(=O)C1=CC=CC=C1")
linkers_input = st.text_area("Linkers (one per line)", value="CCOC(=O)\nCCN(CC)CC\nCCOC(=O)N(C)C")
min_qed = st.slider("Min QED", 0.0, 1.0, 0.3, step=0.05)
max_logp = st.slider("Max LogP", 0.0, 10.0, 5.0, step=0.1)
max_molwt = st.slider("Max MolWt", 100.0, 1000.0, 500.0, step=10.0)

results = []

if st.button("Generate ADCs"):
    linkers = [l.strip() for l in linkers_input.splitlines() if l.strip()]
    for i, linker in enumerate(linkers, 1):
        full_smiles = warhead + linker + payload
        mol = Chem.MolFromSmiles(full_smiles)
        if mol:
            qed = Descriptors.qed(mol)
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)

            if qed < min_qed or logp > max_logp or mw > max_molwt:
                continue

            st.subheader(f"ADC {i}")
            st.image(mol_to_image(mol), caption="2D Structure", use_column_width=False)

            mol3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3d)
            AllChem.UFFOptimizeMolecule(mol3d)
            html_view = show_3d_molecule(mol3d)
            components.html(html_view, height=400)

            cost, timeline, risk = mock_simulation(qed, logp, mw)
            stability = predict_stability(mol)
            release = predict_release_profile(logp, mw)
            similar_name, similarity = find_similar_linker(linker)

            if similar_name in public_linkers:
                st.markdown(f"**Closest Known Linker:** {similar_name} (Similarity Score: {similarity})")
                similar_mol = Chem.MolFromSmiles(public_linkers[similar_name])
                st.image(mol_to_image(similar_mol), caption=f"{similar_name} Structure", use_column_width=False)

            results.append({
                "ADC": f"ADC {i}",
                "SMILES": full_smiles,
                "QED": round(qed, 3),
                "LogP": round(logp, 3),
                "MolWt": round(mw, 1),
                "Cost ($)": cost,
                "Timeline (days)": timeline,
                "Failure Risk": risk,
                "Plasma Stability": stability,
                "Release Profile": release,
                "Closest Known Linker": similar_name,
                "Similarity Score": similarity
            })
        else:
            st.warning(f"Invalid SMILES for linker {i}: {linker}")

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
