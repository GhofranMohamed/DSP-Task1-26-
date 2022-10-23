from tkinter import HORIZONTAL, Menu
from turtle import width
import streamlit as st 
from streamlit_option_menu import option_menu
from scipy.interpolate import CubicSpline
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt, mpld3

st.set_option('deprecation.showPyplotGlobalUse', False)
# button =st.button("Start the porgramer")
# if button :
#     st.session_state['Choose']=1
# #starting a session
# if 'Choose' not in st.session_state :
#     st.session_state['Choose']=0    

# elif 'Choose' in st.session_state==1 :
    #getting maximum frequecny of the signal
def get_max_freq(magnitude=[],time=[]):
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes=np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 0.001:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    max_freq = max(fft_clean_frequencies_array)
    return max_freq 

#comopse page
def composes ():
        try:
            
            fig1 = plt.figure()
            plt.xlabel("Time")
            plt.ylabel("Amplitude")

            frequency= st.sidebar.number_input("Add Frequency",min_value=0,max_value=1000,value=0)
            amplitude= st.sidebar.number_input("Add Amplitude",min_value=0,max_value=1000,value=0)
            time= np.linspace(0,2,1000)
        
        # signal = amplitude*np.sin(2*np.pi*frequency*time)

        

        # st.line_chart(signal)
        
        ####session_state
            if 'signal' not in st.session_state:
                st.session_state.signal =np.sin(0)

            signal_name=st.sidebar.text_input("Enter signal name")   
            add = st.sidebar.button('Add')
            if add:
                st.session_state.signal += amplitude*np.sin(2*np.pi*frequency*time)
            
            st.sidebar.multiselect("Added signals",[signal_name])    
            remove = st.sidebar.button('Remove')
            if remove:
                st.session_state.signal -= amplitude*np.sin(2*np.pi*frequency*time)
            #  st.multiselect("Added signals",[signal_name])


            st.line_chart(st.session_state.signal) 
        # print(st.session_state.signal)
        except Exception as e:
            st.title("Enter your signal")


#draw the signal and interpolate the points
def draw ():
    if  'uploaded' not in st.session_state  :
        uploaded_file = st.file_uploader(
        label="Upload your  file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            global df
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                df = pd.read_excel(uploaded_file)
            st.session_state.uploaded =df
    
    if  'uploaded' in st.session_state  :
        
            sample_factor = st.slider(label="Sample Frequency" , min_value=1, max_value=10, step=1)
            time = st.session_state.uploaded.iloc[:,0]
            amplitude=st.session_state.uploaded.iloc[:,1]
            f_max=get_max_freq(amplitude,time)
            f_samples = sample_factor * f_max
            samples_x_coor=[]
            samples_y_coor=[]
            counter=0
            while(counter<len(time)):
                samples_x_coor.append(time[counter])
                samples_y_coor.append(amplitude[counter])
                counter =counter+(len(time)//(f_samples))
            data ={'samples_x_coor':samples_x_coor ,'samples_y_coor':samples_y_coor}
            df2=pd.DataFrame(data)
            fig = px.line(st.session_state.uploaded, x='Time', y='Amplitude')
            fig.add_scatter(x=df2['samples_x_coor'] , y=df2['samples_y_coor'], mode='markers')
            st.plotly_chart(fig, use_container_width=True)
            cubic_interpolation = CubicSpline(samples_x_coor, samples_y_coor, bc_type='natural')
            x_new = np.linspace(samples_x_coor[0], samples_x_coor[-1], 100)
            y_new = cubic_interpolation (x_new)
            plt.figure(figsize = (7,4))
            plt.plot(x_new, y_new, 'b')
            plt.plot(samples_x_coor,samples_y_coor, 'ro')
            plt.title('samples Interpolation')
            plt.xlabel('samples_x_coor')
            plt.ylabel('samples_y_coor')
            plt.show()
            st.pyplot(plt.show())


        # st.line_chart(df)
            st.write("Please upload the signal")
            st.write("you are in  sampling page")

selected= option_menu(
    menu_title=None,
    options=["Sampler","Composer"],
    icons=["house","wifi channel"],
    default_index=0,
    orientation=HORIZONTAL
)
# when selecting Sampler from the navbar
if selected=="Sampler":

    draw()
    
if selected=="Composer":
    composes()