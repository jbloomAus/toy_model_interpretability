import streamlit as st 
from streamlit_option_menu import option_menu
from PIL import Image
import torch
import uuid
import sys
import os
import plotly.express as px
st.set_page_config(layout="wide", page_title="Neural Network Embeddings", page_icon="resources/favicon.ico")

from adam_jermyn.plot_helper import training_plot, sfa_plot, plot_mono_sweep

with st.sidebar:
    background = Image.open("resources/toy_model_interpretability_logo.png")
    st.image(background, use_column_width=True)
    st.title("Toy Model Interpretability")
    st.subheader("Joseph Bloom presenting/reproducing locally the work of Adam Jermyn, Nicholas Schiefer and Evan Hubinger")

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

        
page_selection = option_menu(None, ["Learning Rate", "Initial Bias", "Gelu/Relu"], 
    icons=['book', 'dash', "bezier2"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if page_selection == "Learning Rate":
    col1, col2, col3, col4 = st.columns([0.1,0.5,0.5,0.1])

    with col2:
        l1_outputs = torch.load('hubinger_2022_data/lr1.pt')

        st.subheader("LR1: A higher learning rate can increase monosemanticity and achieve low Loss (Figure 3)")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=Variable, Decay Rate=0, Bias Offset=0, L1 Reg.=0")
        st.write(l1_outputs["outputs"][0]["setup"])
        fig = training_plot(l1_outputs["outputs"], l1_outputs["sweep_var"], 'loss')
        st.pyplot(fig)

        if st.checkbox("Show single feature activations"):
            fig = sfa_plot(l1_outputs["outputs"], l1_outputs["sweep_var"], js = [0,2,5])
            st.pyplot(fig)

    with col3:
        l2_outputs = torch.load('hubinger_2022_data/lr2.pt')

        st.subheader("LR2: Same as LR1 but power-law sampling of features (Figure 6)")
        st.write("Decoder, ReLU, **Power Law**, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=Variable, Decay Rate=0, Bias Offset=0, L1 Reg.=0")
        st.write(l2_outputs["outputs"][0]["setup"])
        fig = training_plot(l2_outputs["outputs"], l2_outputs["sweep_var"], 'loss')
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key="sdgf"):
            fig = sfa_plot(l2_outputs["outputs"], l2_outputs["sweep_var"], js = [0,2,5])
            st.pyplot(fig)
        

    col1, col2, col3, col4 = st.columns([0.1,0.5,0.5,0.1])

    with col2:

        st.subheader("LR3: Setting a negative initial bias with bias decay encourages monosemanticity (Figure 7)")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=Variable, **Decay Rate=0.03, Bias Offset= -1**, L1 Reg.=0")

        l3_outputs = torch.load('hubinger_2022_data/lr3.pt')
        st.write(l3_outputs["outputs"][0]["setup"])
        fig = training_plot(l3_outputs["outputs"], l3_outputs["sweep_var"], log_color=False)
        st.pyplot(fig)
        
        st.error("This experiment appears to have failed. One or two runs appear to have had an exploding gradient.")
        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(l3_outputs["outputs"], l3_outputs["sweep_var"], js = [0,2,5])
            st.pyplot(fig)

    with col3:
        st.subheader("LR4: Like LR3 but with a Power Law Distribution of Features (Figure 12)")
        st.write("Decoder, ReLU, **Power-Law**, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, **Learning Rate=0.03, Decay Rate=-1**, Bias Offset=0, L1 Reg.=0")

        l4_outputs = torch.load('hubinger_2022_data/lr4.pt')
        st.write(l4_outputs["outputs"][0]["setup"])
        fig = training_plot(l4_outputs["outputs"], l4_outputs["sweep_var"], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(l4_outputs["outputs"], l4_outputs["sweep_var"], js = [0,2,5])
            st.pyplot(fig)

if page_selection == "Initial Bias":
    col1, col2, col3, col4 = st.columns([0.1,0.5,0.5,0.1])

    with col2:
        st.subheader("B1: ")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/16$, Learning Rate=0.003, Decay Rate=0.03, Bias Offset=**Variable**, L1 Reg.=0")
        
        b1_outputs = torch.load('hubinger_2022_data/b1.pt')
        st.write(b1_outputs["outputs"][0]["setup"])

        fig = training_plot(b1_outputs["outputs"], b1_outputs["sweep_var"], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(b1_outputs["outputs"], b1_outputs["sweep_var"], js = [0,2,5])
            st.pyplot(fig)

    with col3:
        st.subheader("B2: ")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/32$, Learning Rate=0.003, Decay Rate=0.003, Bias Offset=**Variable**, L1 Reg.=0")
        b2_outputs = torch.load('hubinger_2022_data/b2.pt')
        st.write(b2_outputs["outputs"][0]["setup"])
        fig = training_plot(b2_outputs["outputs"],  b2_outputs["sweep_var"], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(b2_outputs["outputs"],  b2_outputs["sweep_var"], js = [0,2,5])
            st.pyplot(fig)

    with col2: 
        st.subheader("B3: ")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=0.003, Decay Rate=0.003, Bias Offset=**Variable**, L1 Reg.=0")

        b3_outputs = torch.load('hubinger_2022_data/b3.pt')
        st.write(b3_outputs["outputs"][0]["setup"])
        fig = training_plot(b3_outputs["outputs"], b3_outputs['sweep_var'], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(b3_outputs["outputs"], b3_outputs['sweep_var'], js = [0,2,5])
            st.pyplot(fig)

    with col3:
        st.subheader("B4: ")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/128$, Learning Rate=0.003, Decay Rate=0.003, Bias Offset=**Variable**, L1 Reg.=0")
        
        b4_outputs = torch.load('hubinger_2022_data/b4.pt')
        st.write(b4_outputs["outputs"][0]["setup"])
        fig = training_plot(b4_outputs["outputs"], b4_outputs['sweep_var'], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(b4_outputs["outputs"], b4_outputs['sweep_var'], js = [0,2,5])
            st.pyplot(fig)

    with col2:
        st.subheader("B5: ")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/256$, Learning Rate=0.003, Decay Rate=0.003, Bias Offset=**Variable**, L1 Reg.=0")
        
        b5_outputs = torch.load('hubinger_2022_data/b5.pt')
        st.write(b5_outputs["outputs"][0]["setup"])
        fig = training_plot(b5_outputs["outputs"], b5_outputs['sweep_var'], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(b5_outputs["outputs"], b5_outputs['sweep_var'], js = [0,2,5])
            st.pyplot(fig)


if page_selection == "Gelu/Relu":
    col1, col2, col3 = st.columns([0.1,1,0.1])

    with col2:
        st.subheader("B3 ReLU vs B3 GeLU")
        g3_outputs = torch.load('hubinger_2022_data/g3.pt')
        #b3 = torch.load('hubinger_2022_data/b3.pt')

        st.subheader("G3: ")
        st.write("Decoder, GeLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=0.003, Decay Rate=0.003, Bias Offset=**Variable**, L1 Reg.=0")

    col1, col2, col3, col4 = st.columns([0.1,0.5, 0.5,0.1])

    with col2:
        fig = plot_mono_sweep(g3_outputs["outputs"], g3_outputs['sweep_var'])
        st.pyplot(fig)

    with col3:

        g3_outputs = torch.load('hubinger_2022_data/g3.pt')
        st.write(g3_outputs["outputs"][0]["setup"])
        fig = training_plot(g3_outputs["outputs"], g3_outputs['sweep_var'], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(g3_outputs["outputs"], g3_outputs['sweep_var'], js = [0,2,5])
            st.pyplot(fig)

    col1, col2, col3 = st.columns([0.1,1,0.1])

    with col2:
        st.subheader("B3: ")
        st.write("Decoder, ReLU, Uniform, $N=512$, $m=64$, $k=1024$, $\epsilon=1/64$, Learning Rate=0.003, Decay Rate=0.003, Bias Offset=**Variable**, L1 Reg.=0")

    col1, col2, col3, col4 = st.columns([0.1,0.5, 0.5,0.1])
    with col2:
        b3_outputs = torch.load('hubinger_2022_data/b3.pt')
        fig = plot_mono_sweep(b3_outputs["outputs"], b3_outputs['sweep_var'])
        st.pyplot(fig)
    
    with col3:
        st.write(b3_outputs["outputs"][0]["setup"])
        fig = training_plot(b3_outputs["outputs"], b3_outputs['sweep_var'], log_color=False)
        st.pyplot(fig)

        if st.checkbox("Show single feature activations", key = uuid.uuid4()):
            fig = sfa_plot(b3_outputs["outputs"], b3_outputs['sweep_var'], js = [0,2,5])
            st.pyplot(fig)

