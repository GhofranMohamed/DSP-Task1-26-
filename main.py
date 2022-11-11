from email.policy import default
from logging import exception
from re import I
from tkinter import HORIZONTAL, Menu
from turtle import width
import streamlit as st
from streamlit_option_menu import option_menu
from scipy.interpolate import CubicSpline
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

def draw():
    if 'uploaded' not in st.session_state:
        uploaded_file = st.sidebar.file_uploader(
            label="Upload your  file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            global df
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                df = pd.read_excel(uploaded_file)
            st.session_state.uploaded = df

    if 'uploaded' in st.session_state:

        sample_factor = st.sidebar.slider(
            label="Sample Frequency", min_value=1, max_value=10, step=1)
        time = st.session_state.uploaded.iloc[:, 0]
        amplitude = st.session_state.uploaded.iloc[:, 1]
        f_max = get_max_freq(amplitude, time)
        f_samples = sample_factor * f_max

        noise_check_box = st.sidebar.checkbox(label="Add Noise", value=False)

        if noise_check_box:
            snr_slider = st.sidebar.slider(
                label="SNR", min_value=1, max_value=100, step=1)
            noised_signal = noise(st.session_state.uploaded, snr_slider)
            amplitude_noised = noised_signal.iloc[:, 1]
            time_noised = noised_signal.iloc[:, 0]
            f_max = get_max_freq(amplitude_noised, time_noised)
            f_samples = sample_factor * f_max
            samples_noised_x_coor = []
            samples_noised_y_coor = []
            counter = 0
            while(counter < len(time)):
                samples_noised_x_coor.append(time[counter])
                samples_noised_y_coor.append(amplitude_noised[counter])
                counter = counter+(len(time)//(f_samples))
            data1 = {'samples_noised_x_coor': samples_noised_x_coor,
                     'samples_noised_y_coor': samples_noised_y_coor}
            df3 = pd.DataFrame(data1)
            noised_fig = px.line(noised_signal, x='time',
                                 y='amplitude', title="The noised signal")
            noised_fig.add_scatter(
                x=df3['samples_noised_x_coor'], y=df3['samples_noised_y_coor'], mode='markers')
            noised_fig.update_layout(font_size=23)
            st.plotly_chart(noised_fig, use_container_width=True)
            time = st.session_state.uploaded.iloc[:, 0]
            sampled_amplitude = df3.iloc[:, 1]
            sampled_time = df3.iloc[:, 0]
            T = (sampled_time[1] - sampled_time[0])
            sincM = np.tile(time, (len(sampled_time), 1)) - \
                np.tile(sampled_time[:, np.newaxis], (1, len(time)))
            yNew = np.dot(sampled_amplitude, np.sinc(sincM/T))
            fig = plt.figure()
            plt.plot(time, yNew, label="Reconstructed Signal")
            # plt.scatter(
            #     sampled_time, sampled_amplitude, color='r', label="Sampling Points", marker='x')
            fig.legend()
            plt.title("Reconstructed Signal")
            st.plotly_chart(fig, use_container_width=True)
            # cubic_interpolation = CubicSpline(
            #     samples_x_coor, samples_y_coor, bc_type='natural')
            # x_new = np.linspace(samples_x_coor[0], samples_x_coor[-1], 100)
            # y_new = cubic_interpolation(x_new)
            # fig_4 = plt.figure(figsize=(7, 4))
            # plt.plot(x_new, y_new, 'b')

            # plt.plot(samples_x_coor, samples_y_coor, 'ro')
            # plt.title('samples Interpolation')
            # plt.xlabel('samples_x_coor')
            # plt.ylabel('samples_y_coor')
            # fig_4.legend()
            # # plt.show()
            # st.plotly_chart(fig_4, use_container_width=True)

        if not noise_check_box:
            samples_x_coor = []
            samples_y_coor = []
            counter = 0
            while(counter < len(time)):
                samples_x_coor.append(time[counter])
                samples_y_coor.append(amplitude[counter])
                counter = counter+(len(time)//(f_samples))

            data = {'samples_x_coor': samples_x_coor,
                    'samples_y_coor': samples_y_coor}
            df2 = pd.DataFrame(data)
            time = st.session_state.uploaded.iloc[:, 0]
            sampled_amplitude = df2.iloc[:, 1]
            sampled_time = df2.iloc[:, 0]
            T = (sampled_time[1] - sampled_time[0])
            sincM = np.tile(time, (len(sampled_time), 1)) - \
                np.tile(sampled_time[:, np.newaxis], (1, len(time)))
            yNew = np.dot(sampled_amplitude, np.sinc(sincM/T))
            fig_uploaded_sig = px.line(
                st.session_state.uploaded, x='Time', y='Amplitude')
            fig_uploaded_sig.add_scatter(x=df2['samples_x_coor'],
                                        y=df2['samples_y_coor'], mode='markers')
            fig_uploaded_sig.update_layout(font_size=23)
            st.plotly_chart(fig_uploaded_sig, use_container_width=True)
            # plt.subplot(212)
            fig = plt.figure()
            plt.plot(time, yNew, label="Reconstructed Signal")
            # plt.scatter(
            #     sampled_time, sampled_amplitude, color='r', label="Sampling Points", marker='x')
            fig.legend()
            plt.title("Reconstructed Signal")
            st.plotly_chart(fig, use_container_width=True)
        return st.session_state.uploaded

def get_max_freq(magnitude=[], time=[]):
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes = np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 0.001:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    max_freq = max(fft_clean_frequencies_array)
    return max_freq

# noise function


def noise(data, snr_db):
    signal = data.iloc[:, 1]
    power = (signal)**2
    # signal_power_db = 10 * np.log10(power)
    signal_avg_power = np.mean(power)
    signal_avg_power_db = 10 * np.log10(signal_avg_power)
    noise_db = signal_avg_power_db - snr_db
    noise_watts = 10**(noise_db/10)
    signal_noise = np.random.normal(0, np.sqrt(noise_watts), len(signal))
    noised_signal = signal + signal_noise
    noised = pd.DataFrame(
        {"time": data.iloc[:, 0], "amplitude": noised_signal})
    return noised


# comopse page
def composes():
    if 'signal' not in st.session_state:
            st.session_state.signal = np.sin(0)
            st.session_state.signals = []
            st.session_state.sig_name = []
    # try:
    check_box=st.sidebar.radio(label="Original Signal",options=("Orignal_signal","Generated_signal"))
    if check_box == "Orignal_signal" :
        
        draw()
        
    elif check_box=="Generated_signal":
        
        frequency = st.sidebar.slider(
            "Add Frequency", min_value=0, max_value=10, value=4)
        amplitude = st.sidebar.slider(
            "Add Amplitude", min_value=0, max_value=10, value=4)
        phase = st.sidebar.selectbox(
            label="Choose the wave type", options=("sin", "cos"))
        time = np.linspace(0, 2, 500)
        signal_name = st.sidebar.text_input("Enter signal name")
        
        if phase == "sin":
            signal = amplitude*np.sin(2*np.pi*frequency*time)
        elif phase == "cos":
            signal = amplitude*np.cos(2*np.pi*frequency*time)
        fig = px.line(signal)
        st.plotly_chart(fig, use_container_width=True)
    add = st.sidebar.button('Add')
    
    if add and check_box == "Orignal_signal":
        amplitude_original = st.session_state.uploaded.iloc[:, 1]
        added_values = ["original signal", 1 ,amplitude_original]
        st.session_state.signals.append(added_values)
        st.session_state.sig_name.append("original signal")
        st.session_state.signal += amplitude_original
    if add and check_box == "Generated_signal":
        added_values = [signal_name, frequency, amplitude]  
        st.session_state.signals.append(added_values)
        st.session_state.sig_name.append(signal_name)
        if frequency == "sin":
            st.session_state.signal += amplitude*np.sin(2*np.pi*frequency*time)
        else :
            st.session_state.signal += amplitude*np.cos(2*np.pi*frequency*time)
           

    if st.session_state.sig_name is not None:
        removed_signal = st.sidebar.selectbox(
            "Added signals", st.session_state.sig_name)
        remove = st.sidebar.button('Remove')
        if remove and check_box == "Orignal_signal":
            i = 0
            for sig in st.session_state.signals:
                if removed_signal == sig[0]:
                    st.session_state.signal -= sig[1]
                    st.session_state.signals.pop(i)
                    st.session_state.sig_name.remove(removed_signal)

                    break
                i += 1
        if remove and check_box == "Generated_signal":
            i = 0
            for sig in st.session_state.signals:
                if removed_signal == sig[0]:
                    if frequency == "sin":
                        st.session_state.signal -= sig[2] * \
                        np.sin(2*np.pi*sig[1]*time)                    
                    else :
                        st.session_state.signal -= sig[2] * \
                        np.cos(2*np.pi*sig[1]*time)

                    st.session_state.signals.pop(i)
                    st.session_state.sig_name.remove(removed_signal)

                    break
                i += 1

    saved_sig = st.sidebar.text_input("Name the signal to save")
    save_bt = st.sidebar.button(label="Save")
    if save_bt:
        save_file(saved_sig)
    try :
        if st.session_state.signal is not None:
            st.line_chart(st.session_state.signal)
    except Exception as e :
        print(st.session_state.signal)

        # fig1 = px.line(st.session_state.signal)
        # st.plotly_chart(fig1, use_container_width=True)

        # print(st.session_state.sigal) 
    #  st.multiselect("Added signals",[signal_name])

    # print(st.session_state.signal)

    # except Exception as e:


# draw the signal and interpolate the points


def save_file(name):
    final_signal = pd.DataFrame(st.session_state.signal)
    final_signal.to_csv("%s.csv" % name, index=False)


composes()
    