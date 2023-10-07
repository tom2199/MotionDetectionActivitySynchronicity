from scripts.preprocessing import *
import os
import plotly.graph_objects as go
import plotly
import pandas as pd
from scipy import signal, stats, fftpack


# use preprocessing.py to import data
def bin_to_dataframe(file):
    data = readBinFile(file)
    dataframe = decompress(data)
    return dataframe


# use preprocessing.py to make dataframes equidistant
def make_dataframes_equidistant(dataframe_list, sampling_rate):
    output = []
    dataframe_equidistant_list = make_equidistant(dataframe_list, sampling_rate)[0]
    for dataframe in dataframe_equidistant_list:
        output.append(resample_raw_data(dataframe, sampling_rate))
    return output


# import data from specified directory and list of mac addresses and output a dictionary of dataframes
def import_folder(mac_address_list, folder):
    output_dict = {}
    for mac_address in mac_address_list:
        data = []
        for file in os.listdir(folder):
            if file.startswith(mac_address + "_d") & file.endswith(".bin"):
                data.append(bin_to_dataframe(folder + "/" + file))
        if len(data) > 0:
            data.sort(key=lambda d: d.index[0])
            data = make_dataframes_equidistant(data, 25)
            output_dict[mac_address] = data
        else:
            print("No data found for " + mac_address)

    return output_dict


# put all dataframes of a list into a list of traces for plotly
def trace_dataframe_list(dataframe_list, sensor_names, showlegend=True):
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    output = []
    for dataframe in dataframe_list:
        y_axis = dataframe[sensor_names]
        x_axis = dataframe.index
        for index, s in enumerate(sensor_names):
            output.append(go.Scattergl(x=x_axis, y=y_axis[s], name=s, legendgroup=s, showlegend=showlegend,
                                       line=dict(color=colors[index])))
    return output


# return a list of dataframes within the specified time range
def return_dataframe_list_within_timerange(dataframe_list, from_time, to_time):
    output = []
    for dataframe in dataframe_list:
        if ((dataframe.index[len(dataframe.index) - 1] > pd.Timestamp(from_time)) & (
                dataframe.index[0] < pd.Timestamp(to_time))):
            output.append(dataframe)
    return output


# calculate vector magnitude
def calc_vec_mag(dataframe):
    return np.sqrt(np.square(dataframe).sum(axis=1))


# calculate point where cross-correlation is highest
def calc_lag(a, b):
    corr = signal.correlate(a, b)
    lags = signal.correlation_lags(len(a), len(b))
    return int(lags[np.where(corr == np.max(corr))[0]][0])


# calculate the selected attributes in the selected timeframe
def calc_rolling_attributes(dataframe, attributes=None, timeframe="10s", iqr=None):
    if iqr is None:
        iqr = [0.25, 0.75]
    if attributes is None:
        attributes = ["min", "max", "mean", "std", "quant0.1", "quant0.3", "quant0.5", "quant0.7",
                      "quant0.9", "fft_freq", "diff", "iqr0.1", "iqr0.2", "iqr0.4", "iqr0.6", "iqr0.8", "iqr0.9"]
    if "quant_all" in attributes:
        attributes.remove("quant_all")
        attributes.append("quant0.1")
        attributes.append("quant0.3")
        attributes.append("quant0.5")
        attributes.append("quant0.7")
        attributes.append("quant0.9")
    if "iqr_all" in attributes:
        attributes.remove("iqr_all")
        attributes.append("iqr0.1")
        attributes.append("iqr0.2")
        attributes.append("iqr0.4")
        attributes.append("iqr0.6")
        attributes.append("iqr0.8")
        attributes.append("iqr0.9")
    output = pd.DataFrame()

    if "min" in attributes:
        output["min"] = dataframe.rolling(timeframe).min().resample(timeframe).first()
    if "max" in attributes:
        output["max"] = dataframe.rolling(timeframe).max().resample(timeframe).first()
    if "mean" in attributes:
        output["mean"] = dataframe.rolling(timeframe).mean().resample(timeframe).first()
    if "std" in attributes:
        output["std"] = dataframe.rolling(timeframe).std().resample(timeframe).first()
    if "quant0.1" in attributes:
        output["quant0.1"] = dataframe.rolling(timeframe).quantile(0.1).resample(timeframe).first()
    if "quant0.3" in attributes:
        output["quant0.3"] = dataframe.rolling(timeframe).quantile(0.3).resample(timeframe).first()
    if "quant0.5" in attributes:
        output["quant0.5"] = dataframe.rolling(timeframe).quantile(0.5).resample(timeframe).first()
    if "quant0.7" in attributes:
        output["quant0.7"] = dataframe.rolling(timeframe).quantile(0.7).resample(timeframe).first()
    if "quant0.9" in attributes:
        output["quant0.9"] = dataframe.rolling(timeframe).quantile(0.9).resample(timeframe).first()

    if "fft_freq" in attributes and len(attributes) > 1:
        temp = []
        for index in output.index:
            if index == output.index[-1]:
                temp.append(0.0)
            else:
                fft_freq = calc_fft_freq(dataframe[index:(index + pd.Timedelta(timeframe))])
                temp.append(fft_freq)
        output["fft_freq"] = temp

    if "min" and "max" and "diff" in attributes:
        output["diff"] = output["max"] - output["min"]

    if "iqr" in attributes:
        output["iqr"] = calc_iqr(dataframe, iqr[0], iqr[1], timeframe=timeframe)

    if "iqr0.1" in attributes:
        output["iqr0.1"] = calc_iqr(dataframe, 0.45, 0.55, timeframe=timeframe)
    if "iqr0.2" in attributes:
        output["iqr0.2"] = calc_iqr(dataframe, 0.4, 0.6, timeframe=timeframe)
    if "iqr0.4" in attributes:
        output["iqr0.4"] = calc_iqr(dataframe, 0.3, 0.7, timeframe=timeframe)
    if "iqr0.6" in attributes:
        output["iqr0.6"] = calc_iqr(dataframe, 0.2, 0.8, timeframe=timeframe)
    if "iqr0.8" in attributes:
        output["iqr0.8"] = calc_iqr(dataframe, 0.1, 0.9, timeframe=timeframe)
    if "iqr0.9" in attributes:
        output["iqr0.9"] = calc_iqr(dataframe, 0.05, 0.95, timeframe=timeframe)

    return output


# calculate rolling correlation within selected window
def calc_rolling_corr(a, b, window=30, hmean=True, scatter=True):
    rolling_corr = a.rolling(window).corr(b)
    rolling_corr = (rolling_corr.dropna() + 1) / 2
    if hmean:
        rolling_corr["hmean"] = stats.hmean(rolling_corr, axis=1)
    if scatter:
        rolling_corr["scatter"] = rolling_corr.max(axis=1) - rolling_corr.min(axis=1)
    return rolling_corr


# calculate the prevalent frequency of the dataframe via fft
def calc_fft_freq(dataframe, cutoff=0.1, hz=25):
    fft = fftpack.fft(dataframe.values)
    psd = np.abs(fft) ** 2
    fftfreq = fftpack.fftfreq(len(psd), 1. / hz)
    i = fftfreq > cutoff
    return float(fftfreq[np.where(psd[i] == psd[i].max())][0])


# remove outliers in the dataset using the Z-Score
def remove_outliers_by_zscore(dataframe_column, threshold=3, timeframe="10s"):
    z = np.abs(stats.zscore(dataframe_column))
    output = pd.DataFrame()
    output = dataframe_column[z < threshold]
    return output.resample(timeframe).interpolate()


# calculate IQR within timeframe
def calc_iqr(dataframe, lower_quant=0.25, upper_quant=0.75, timeframe="10s"):
    low = dataframe.rolling(timeframe).quantile(lower_quant).resample(timeframe).first()
    high = dataframe.rolling(timeframe).quantile(upper_quant).resample(timeframe).first()
    return (high - low).abs()


# calculate activity based on the difference between min and max
def calc_activity(df_diff_columm, diff_treshold=1, rolling_treshold=4):
    activity = pd.DataFrame()
    activity = df_diff_columm > diff_treshold
    activity = activity.rolling(6).sum()
    activity.loc[activity < rolling_treshold] = 0
    activity.loc[activity >= rolling_treshold] = 1
    return activity


# calculate score based on quantile start and stop treshold
def calc_score(df_iqr_column, start_treshold=0.95, stop_treshold=0.75, resample=None, timeframe=None):
    output = pd.DataFrame(index=df_iqr_column.index)
    activity_started = False
    score = []
    for row in df_iqr_column:
        iqr = row
        if iqr > start_treshold:
            if not activity_started:
                activity_started = True
        elif iqr < stop_treshold:
            if activity_started:
                activity_started = False

        if activity_started:
            score.append(1)
        else:
            score.append(0)
    output[""] = score
    if resample is not None and timeframe is not None:
        output = output.resample(resample).mean().resample(timeframe).interpolate()
    return output
