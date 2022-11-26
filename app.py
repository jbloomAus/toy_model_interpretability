import streamlit as st 
from PIL import Image
import torch
import sys
import os
import plotly.express as px
st.set_page_config(layout="wide", page_title="Neural Network Embeddings", page_icon="resources/favicon.ico")

from adam_jermyn.plot_helper import training_plot, sfa_plot

with st.sidebar:
    background = Image.open("resources/toy_model_interpretability_logo.png")
    st.image(background, use_column_width=True)
    st.title("Toy Model Interpretability")
    st.subheader("Very small additions by Joseph Bloom building on work by Adam Jermyn, Nicholas Schiefer and Evan Hubinger")

    """
    This repo is mostly based on the work of Adam, Evan and Nicholas in their [paper](https://arxiv.org/abs/2211.09169). 
    You can find their original work as a submodule which I've decided to use as a starting point for my own investigations. 
    For personal taste, I'm going to restructure their code into a code-base with modularity/tests. 

    """

    #st.image("resources/toy_model_interpretability_logo.png")

col1, col2, col3 = st.columns([0.1,1,0.1])

with col2:
    st.title("Working with the code from: 'Engineering Monosemanticity in Toy Models'")

    with st.expander("Batch Reference"):
        '''
        Table 1: Training parameters and model architectures.

        | Batch | Task | Activation | Feature Dist. | $N$ | $m$ | $k$ | $\epsilon$ | Learning Rate | Decay Rate | Bias Offset | L1 Reg. |
        | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
        | LR1 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 64$ | Variable | 0 | 0 | 0 |
        | LR2 | Decoder | ReLU | Power-law | 512 | 64 | 1024 | $1 / 64$ | Variable | 0 | 0 | 0 |
        | LR3 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 64$ | Variable | $0.03$ | $-1$ | 0 |
        | B1 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 16$ | $0.003$ | $0.03$ | Variable | 0 |
        | B2 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 32$ | $0.003$ | $0.003$ | Variable | 0 |
        | B3 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 64$ | $0.003$ | $0.003$ | Variable | 0 |
        | B4 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 128$ | $0.003$ | $0.003$ | Variable | 0 |
        | B5 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 256$ | $0.003$ | $0.003$ | Variable | 0 |
        | LR4 | Decoder | ReLU | Power-law | 512 | 64 | 1024 | $1 / 64$ | Variable | $0.03$ | $-1$ | 0 |
        | B3 | Decoder | GeLU | Uniform | 512 | 64 | 1024 | $1 / 64$ | Variable | $0.03$ | Variable | 0 |
        | E1 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | Variable | $0.003$ | $0.03$ | $-1$ | 0 |
        | E2 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | Variable | $0.003$ | $0.01$ | $-1$ | 0 |
        | E3 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | Variable | $0.003$ | $0.003$ | $-1$ | 0 |
        | E4 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | Variable | $0.003$ | $0.001$ | $-1$ | 0 |
        | K0 | Decoder | ReLU | Uniform | 512 | 64 | Variable | $1 / 64$ | $0.007$ | 0 | 0 | 0 |
        | K1 | Decoder | ReLU | Uniform | 512 | 64 | Variable | $1 / 64$ | $0.007$ | $0.03$ |$-1$  | 0 |
        | K2 | Decoder | ReLU | Power-law  | 512 | 64 | Variable | $1 / 64$ | $0.007$ | $0.03$ |$-1$  | 0 |
        | RG1 | Decoder | ReLU | Uniform | 512 | 64 | 1024 | $1 / 64$ | $0.005$ | $0.03$ | $-1$  | Variable |
        | RP1 | Re-Projector | ReLU | Uniform | 512 | 64 | 1024 | $1 / 64$ | Variable | $0.03$ | $-1$  | 0 |
        | LR5 | Abs | ReLU | Uniform | 512 | 64 | 2048 | $1 / 64$ | Variable | $0.03$ | $-1$  | 0 |
        | D1 | Abs | ReLU | Uniform | 512 | 64 | 2048 | $1 / 64$ | 0.007 | Variable | $-1$  | 0 |
        '''

col1, col2, col3, col4 = st.columns([0.1,0.5,0.5,0.1])

with col2:
    l1_outputs = torch.load('hubinger_2022_data/lr1.pt')

    st.subheader("LR1: A higher learning rate can increase monosemanticity and achieve low Loss (Figure 3)")
    st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=Variable, Decay Rate=0, Bias Offset=0, L1 Reg.=0")

    fig = training_plot(l1_outputs["outputs"], l1_outputs["sweep_var"], 'loss')
    st.pyplot(fig)

    if st.checkbox("Show single feature activations"):
        fig = sfa_plot(l1_outputs["outputs"], l1_outputs["sweep_var"], js = [0,2,5])
        st.pyplot(fig)

with col3:
    l2_outputs = torch.load('hubinger_2022_data/lr2.pt')

    st.subheader("LR2: Same as LR1 but power-law sampling of features (Figure 6)")
    st.write("Decoder, ReLU, **Power Law**, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=Variable, Decay Rate=0, Bias Offset=0, L1 Reg.=0")

    fig = training_plot(l2_outputs["outputs"], l2_outputs["sweep_var"], 'loss')
    st.pyplot(fig)

    if st.checkbox("Show single feature activations", key="sdgf"):
        fig = sfa_plot(l2_outputs["outputs"], l2_outputs["sweep_var"], js = [0,2,5])
        st.pyplot(fig)
    
col1, col2, col3 = st.columns([0.1,1,0.1])

with col2:
    st.subheader("LR3: Setting a negative initial bias with bias decay encourages monosemanticity (Figure 7)")
    st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=Variable, **Decay Rate=0.03, Bias Offset= -1**, L1 Reg.=0")


col1, col2, col3, col4 = st.columns([0.1,0.5,0.5,0.1])

with col2:

    l3_outputs = torch.load('hubinger_2022_data/lr3.pt')

    fig = training_plot(l3_outputs["outputs"], l3_outputs["sweep_var"], 'loss')
    st.pyplot(fig)
    
    st.error("This experiment appears to have failed. One or two runs appear to have had an exploding gradient.")
    if st.checkbox("Show single feature activations", key = "sdgasgas"):
        fig = sfa_plot(l3_outputs["outputs"], l3_outputs["sweep_var"], js = [0,2,5])
        st.pyplot(fig)