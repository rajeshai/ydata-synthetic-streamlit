import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_synthetic.synthesizers.regular import DRAGAN, CGAN, CRAMERGAN, WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

st.set_page_config(layout="wide",initial_sidebar_state="auto")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
def run():
    st.sidebar.image('YData_logo.svg')
    st.title('Generate synthetic data for a tabular classification dataset using [ydata-synthetic](https://github.com/ydataai/ydata-synthetic)')
    st.markdown('This streamlit application can generate synthetic data for your dataset. Please read all the instructions in the sidebar before you start the process.')
    data = st.file_uploader('Upload a preprocessed dataset in csv format')
    st.sidebar.title('About')
    st.sidebar.markdown('[ydata-synthetic](https://github.com/ydataai/ydata-synthetic) is an open-source library and is used to generate synthetic data mimicking the real world data.')
    st.sidebar.header('What is synthetic data?')
    st.sidebar.markdown('Synthetic data is artificially generated data that is not collected from real world events. It replicates the statistical components of real data without containing any identifiable information, ensuring individuals privacy.')
    st.sidebar.header('Why Synthetic Data?')
    st.sidebar.markdown('''Synthetic data can be used for many applications:
- Privacy
- Remove bias
- Balance datasets
- Augment datasets''')


    st.sidebar.header('Steps to follow')
    st.sidebar.markdown('''
- Upload any preprocessed tabular classification dataset.
- Choose the parameters in the adjacent window appropriately
- Since this is a demo, please choose less number of epochs for quick completion of training
- After choosing all parameters, Click the button under the parameters to start training.
- After the training is complete, you will see a button to download your synthetic dataset. Click that to download.
- You will also see a button to compare your real dataset with the synthetic dataset that's generated.
- Click that button to get the graphs comparing two datasets.''')

    st.sidebar.markdown('''[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/ydataai/ydata-synthetic)''',unsafe_allow_html=True)

    @st.cache
    def train(df):
        st.write('Model Training is in progress. It may take a few minutes. Please wait for a while.')
        models_dir = './cache'
        gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate*0.001,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=layer_dim)

        train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)
            
        synthesizer = model(gan_args, n_discriminator=3)
        synthesizer.train(data, train_args, num_cols, cat_cols)
        synthesizer.save('data_synth.pkl')
        synthesizer = model.load('data_synth.pkl')
        data_syn = synthesizer.sample(samples)
        return data_syn
    
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    @st.cache
    def download(df):
        if df is not None:
            csv = convert_df(df)
            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='data_syn.csv',
            mime='text/csv')
        else:
            st.write('Please generate a synthetic dataset to download!!')
    
    if data is not None:
        data = pd.read_csv(data)
        data.dropna(inplace=True)
        st.header('Choose the parameters!!')
        col1, col2, col3,col4 = st.columns(4)
        with col1:
            model = st.selectbox('Choose the GAN model', ['DRAGAN','CGAN','CRAMEGAN','WGAN_GP'],key=1)
            if model=='DRAGAN':
                model = DRAGAN
            elif model=='CGAN':
                model=CGAN
            elif model=='CRAMEGAN':
                model = CRAMERGAN
            else:
                model = WGAN_GP
            num_cols = st.multiselect('Choose the numerical columns', data.columns,key=1)
            cat_cols = st.multiselect('Choose categorical columns', [x for x in data.columns if x not in num_cols], key=2)

        with col2:
            noise_dim = st.number_input('Select noise dimension', 0,200,128,1)
            layer_dim = st.number_input('Select the layer dimension', 0,200,128,1)
            batch_size = st.number_input('Select batch size', 0,500, 500,1)

        with col3:
            log_step = st.number_input('Select sample interval', 0,200,100,1)
            epochs = st.number_input('Select the number of epochs',0,50,2,1)
            learning_rate = st.number_input('Select learning rate(x1e-3', 0.01, 0.1, 0.05, 0.01)

        with col4:
            beta_1 = st.slider('Select first beta co-efficient', 0.0, 1.0, 0.5)
            beta_2 = st.slider('Select second beta co-efficient', 0.0, 1.0, 0.9)
            samples = st.number_input('Select the number of synthetic samples to be generated', 0, 400000, step=1000)





    if st.button('Click here to start the training process'):
        if data is not None:
            data_syn = train(data)
            st.success('Synthetic dataset with the given number of samples is generated!!')
        else:
            st.write('Please upload a dataset to train')
        
    if st.button('Click here to download the dataset'):
        if data is not None:
            download(data_syn)
        else:
            st.write('Please generate a synthetic dataset to download!!')
    if st.button('Click here to compare to the two datasets'):
        if data is not None:
            f , axes =  plt.subplots(len(cat_cols),2, figsize=(20,25))
            f.suptitle('Real data vs Synthetic data')
            for i, j in enumerate(cat_cols):
                sns.countplot(x=j, data=data, ax = axes[i,0])
                sns.countplot(x=j, data=data_syn, ax = axes[i,1])
        else:
            st.write('Can not plot the graphs. Please create a synthetic dataset!!')


if __name__== '__main__':
    run()