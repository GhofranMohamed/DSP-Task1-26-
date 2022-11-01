import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


fig1 = go.Figure()
# set x axis label
fig1.update_xaxes(
    title_text="Time (Sec)",  # label
    title_font={"size": 20},
    title_standoff=25)
# set y axis label
fig1.update_yaxes(
    title_text="Amplitude(mv)",
    title_font={"size": 20},
    # label
    title_standoff=25)


st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)
if 'signal' not in st.session_state:
    st.session_state.signal = np.sin(0)
    st.session_state.signals = []
    st.session_state.sig_name = []
with open("style.css")as source_des:
    st.markdown(f"<style>{source_des.read()} </style>", unsafe_allow_html=True)

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        frequency = st.slider(
            "Frequency", min_value=0, max_value=15, value=1)
    with col2:
        amplitude = st.slider(
            "Amplitude", min_value=0, max_value=15, value=1)
    phase = st.selectbox(
        label="Wave type", options=("cos", "sin"))
with st.sidebar:
    col3, col4 = st.columns(2)
    with col3:

        max_frequency = st.number_input(
            label="max_frequency", min_value=1, step=1, value=5)

    with col4:
        sample_factor = st.slider(
            label="Sample Frequency", min_value=1, max_value=5*max_frequency, step=1)

with st.sidebar:
    col7, col8 = st.columns(2)
    with col7:
        signal_name = st.text_input("Enter signal name", value="Untitled")

    with col8:
        removed_signal = st.selectbox(
            "Choose signal ", st.session_state.sig_name)
with st.sidebar:
    col5, col6 = st.columns(2)
    with col5:
        add = st.button('Add')

    with col6:
        remove = st.button('Remove')
with st.sidebar:
    col9, col10 = st.columns(2)
    with col9:
        snr_slider = st.slider(
            label="SNR", min_value=1, max_value=100, step=1)

    with col10:
        saved_sig = st.text_input("Name the signal to save", value="Signal_1")

with st.sidebar:
    col11, col12 = st.columns(2)
    with col11:

        noise_check_box = st.checkbox(label="Add Noise", value=False)

    with col12:
        save_bt = st.button(label="Save")


def convert_to_df():
    if "uploaded" in st.session_state:
        time = st.session_state.uploaded.iloc[:, 0]
        amplitude_upload_file = st.session_state.uploaded.iloc[:, 1]
    else:
        time = np.linspace(0, 5, 1000)
        amplitude_upload_file = np.zeros(len(time))
    mixed_signal = 0
    for sig in st.session_state.signals:
        if sig[3] == "sin":
            mixed_signal += sig[2]*np.sin(2*np.pi*sig[1]*time)
        else:
            mixed_signal += sig[2]*np.cos(2*np.pi*sig[1]*time)
    total_mixed_signal = mixed_signal+amplitude_upload_file
    summated_signal = pd.DataFrame(
        {"time": time, "amplitude": total_mixed_signal})
    return summated_signal


def get_samples_df(data, samples):
    time_samples = data.iloc[:, 0]
    amplitude_samples = data.iloc[:, 1]
    f_samples = samples
    samples_x_coor = []
    samples_y_coor = []
    counter = 0
    while(counter < len(time_samples)):
        samples_x_coor.append(time_samples[counter])
        samples_y_coor.append(amplitude_samples[counter])
        counter = counter+(len(time_samples)//(f_samples))

    data = {'samples_x_coor': samples_x_coor,
            'samples_y_coor': samples_y_coor}
    df_samples = pd.DataFrame(data)
    return df_samples


def get_max_freq(data):
    time = data.iloc[:, 0]
    magnitude = data.iloc[:, 1]
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes = np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 0.001:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    try:
        max_freq = max(fft_clean_frequencies_array)
        return max_freq
    except Exception as e:
        return 1

# noise function


def noise(data, snr_db):
    signal = data.iloc[:, 1]
    power = (signal)**2
    signal_avg_power = np.mean(power)
    signal_avg_power_db = 10 * np.log10(signal_avg_power)
    noise_db = signal_avg_power_db - snr_db
    noise_watts = 10**(noise_db/10)
    signal_noise = np.random.normal(0, np.sqrt(noise_watts), len(signal))
    noised_signal = signal + signal_noise
    noised = pd.DataFrame(
        {"time": data.iloc[:, 0], "amplitude": noised_signal})
    return noised


def reconstruction(df_total, df_samples):
    time = df_total.iloc[:, 0]
    sampled_amplitude = df_samples.iloc[:, 1]
    sampled_time = df_samples.iloc[:, 0]
    try:
        T = (sampled_time[1] - sampled_time[0])
        sincM = np.tile(time, (len(sampled_time), 1)) - \
            np.tile(sampled_time[:, np.newaxis], (1, len(time)))
        yNew = np.dot(sampled_amplitude, np.sinc(sincM/T))
        df_interpolation = pd.DataFrame({'interpolated_x': time,
                                        'interpolated_y': yNew})
        return df_interpolation
    except Exception as e:
        default_signal = pd.DataFrame(
            {"time": [0, 0, 0], "amplitude": [0, 0, 0]})
        return default_signal


def draw(data, data_samples):
    time = data.iloc[:, 0]
    amplitude_summation = data.iloc[:, 1]
    time_samples = data_samples.iloc[:, 0]
    amplitude_samples = data_samples.iloc[:, 1]
    df_reconstructed = reconstruction(data, data_samples)
    reconstructed_time = df_reconstructed.iloc[:, 0]
    reconstructed_amplitude = df_reconstructed.iloc[:, 1]
    fig_sig = fig1.add_scatter(
        x=time, y=amplitude_summation, name="summated signal")
    fig_sig.add_scatter(x=time_samples, y=amplitude_samples,
                        name="samples", mode='markers')
    fig_sig.add_scatter(x=reconstructed_time,
                        y=reconstructed_amplitude, name="reconstructed signal")
    st.plotly_chart(fig_sig, use_container_width=True)


def save_file(name):
    if "uploaded" in st.session_state:
        time = st.session_state.uploaded.iloc[:, 0]
        amplitude_upload_file = st.session_state.uploaded.iloc[:, 1]
    else:
        time = np.linspace(0, 5, 1000)
        amplitude_upload_file = np.zeros(len(time))
    mixed_signal = 0
    for sig in st.session_state.signals:
        if sig[3] == "sin":
            mixed_signal += sig[2]*np.sin(2*np.pi*sig[1]*time)
        else:
            mixed_signal += sig[2]*np.cos(2*np.pi*sig[1]*time)
    total_mixed_signal = mixed_signal+amplitude_upload_file
    summated_signal = pd.DataFrame(
        {"time": time, "amplitude": total_mixed_signal})
    summated_signal.to_csv("%s.csv" % name, index=False)


def main_fun():
    uploaded_file = st.file_uploader(
        label="Upload your  file", type=['csv', 'xlsx'])
    if "signals" not in st.session_state:
        st.session_state.signals = []
        # st.session_state.signals.append(["default", 4, 4, "sin"])
        st.session_state.sig_name = []
        # st.session_state.sig_name.append("default")

    # if uploaded_file is None:
    #     st.session_state.uploaded =pd.DataFrame({"time":[0,0,0],"amplitude":[0,0,0]})

    if uploaded_file is not None:
        st.session_state.uploaded = pd.read_csv(uploaded_file)

    # else :
        # default_signal = pd.DataFrame({"time":[0,0,0],"amplitude":[0,0,0]})

    if add:
        if signal_name is None:
            st.sidebar.error("Please enter the signal name")
        elif signal_name in st.session_state.sig_name:
            st.sidebar.error("Please is already used")
        else:
            added_values_generated = [signal_name, frequency, amplitude, phase]
            st.session_state.signals.append(added_values_generated)
            st.session_state.sig_name.append(signal_name)
            st.experimental_rerun()
    if remove:
        index = 0
        for sig in st.session_state.sig_name:
            if removed_signal == sig:
                st.session_state.signals.pop(index)
                st.session_state.sig_name.pop(index)
                st.experimental_rerun()
                break
            index += 1
    total_data = convert_to_df()
    if not noise_check_box:
        if uploaded_file:
            f_max = get_max_freq(total_data)
        else:
            # st.session_state.uploaded = None
            f_max = max_frequency
        f_samples = f_max*sample_factor
        # f_samples = sample_factor
        samples_df = get_samples_df(total_data, f_samples)
        draw(total_data, samples_df)

    if noise_check_box:
        if uploaded_file:
            f_max = get_max_freq(total_data)
        else:
            # st.session_state.uploaded = None
            f_max = max_frequency
        noised_df = noise(total_data, snr_slider)
        # # f_max = get_max_freq(noised_df)
        # f_max=frequency
        f_samples = f_max*sample_factor
        # f_samples = sample_factor
        samples_df = get_samples_df(noised_df, f_samples)
        draw(noised_df, samples_df)

    if save_bt:
        save_file(saved_sig)


if __name__ == "__main__":
    main_fun()
