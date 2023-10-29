from scripts.preprocessing import *
import os
import plotly.graph_objects as go
import plotly
from scipy import signal, stats, fftpack
import numpy as np

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


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
def trace_dataframe_list(dataframe_list, sensor_names, show_legend=True):
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    colors = colors + colors + colors  # repeat colors to have enough
    output = []
    for df_index, dataframe in enumerate(dataframe_list):
        y_axis = dataframe[sensor_names]
        x_axis = dataframe.index
        for index, s in enumerate(sensor_names):
            if df_index == 0:  # only show legend for first dataframe
                output.append(go.Scattergl(x=x_axis, y=y_axis[s], name=s, legendgroup=s, showlegend=show_legend,
                                           line=dict(color=colors[index])))
            else:
                output.append(go.Scattergl(x=x_axis, y=y_axis[s], name=s, legendgroup=s, showlegend=False,
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
        attributes = ["min", "max", "mean", "std", "quant0.1", "quant0.2", "quant0.3", "quant0.4", "quant0.5",
                      "quant0.6", "quant0.7", "quant0.8", "quant0.9", "fft_freq", "diff", "iqr0.1", "iqr0.2", "iqr0.3",
                      "iqr0.4", "iqr0.5", "iqr0.6", "iqr0.7", "iqr0.8", "iqr0.9"]
    if "quant_all" in attributes:
        attributes.remove("quant_all")
        attributes.append("quant0.1")
        attributes.append("quant0.2")
        attributes.append("quant0.3")
        attributes.append("quant0.4")
        attributes.append("quant0.5")
        attributes.append("quant0.6")
        attributes.append("quant0.7")
        attributes.append("quant0.8")
        attributes.append("quant0.9")
    if "iqr_all" in attributes:
        attributes.remove("iqr_all")
        attributes.append("iqr0.1")
        attributes.append("iqr0.2")
        attributes.append("iqr0.3")
        attributes.append("iqr0.4")
        attributes.append("iqr0.5")
        attributes.append("iqr0.6")
        attributes.append("iqr0.7")
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
    if "quant0.2" in attributes:
        output["quant0.2"] = dataframe.rolling(timeframe).quantile(0.2).resample(timeframe).first()
    if "quant0.3" in attributes:
        output["quant0.3"] = dataframe.rolling(timeframe).quantile(0.3).resample(timeframe).first()
    if "quant0.4" in attributes:
        output["quant0.4"] = dataframe.rolling(timeframe).quantile(0.4).resample(timeframe).first()
    if "quant0.5" in attributes:
        output["quant0.5"] = dataframe.rolling(timeframe).quantile(0.5).resample(timeframe).first()
    if "quant0.6" in attributes:
        output["quant0.6"] = dataframe.rolling(timeframe).quantile(0.6).resample(timeframe).first()
    if "quant0.7" in attributes:
        output["quant0.7"] = dataframe.rolling(timeframe).quantile(0.7).resample(timeframe).first()
    if "quant0.8" in attributes:
        output["quant0.8"] = dataframe.rolling(timeframe).quantile(0.8).resample(timeframe).first()
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
    if "iqr0.3" in attributes:
        output["iqr0.3"] = calc_iqr(dataframe, 0.35, 0.65, timeframe=timeframe)
    if "iqr0.4" in attributes:
        output["iqr0.4"] = calc_iqr(dataframe, 0.3, 0.7, timeframe=timeframe)
    if "iqr0.5" in attributes:
        output["iqr0.5"] = calc_iqr(dataframe, 0.25, 0.75, timeframe=timeframe)
    if "iqr0.6" in attributes:
        output["iqr0.6"] = calc_iqr(dataframe, 0.2, 0.8, timeframe=timeframe)
    if "iqr0.7" in attributes:
        output["iqr0.7"] = calc_iqr(dataframe, 0.15, 0.85, timeframe=timeframe)
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
def calc_activity(df_diff_column, diff_threshold=1, rolling_threshold=4):
    activity = pd.DataFrame()
    activity = df_diff_column > diff_threshold
    activity = activity.rolling(6).sum()
    activity.loc[activity < rolling_threshold] = 0
    activity.loc[activity >= rolling_threshold] = 1
    return activity


# calculate score based on quantile start and stop threshold
def calc_score(df_column, start_threshold=0.95, stop_threshold=0.75, resample=None, timeframe=None):
    output = pd.DataFrame(index=df_column.index)
    activity_started = False
    score = []
    for value in df_column:
        if value > start_threshold:
            if not activity_started:
                activity_started = True
        elif value < stop_threshold:
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


# calculate groups of data where the value is above the threshold and give them an identifier each
def calc_identifier(df_column, gap=30, min_val=0):
    output = pd.DataFrame()
    output = df_column[df_column > min_val]
    output["identifier"] = (~output.index.to_series().diff().dt.seconds.div(gap, fill_value=0).lt(2)).cumsum()
    return output["identifier"]


# calculate window for a single identifier number
def __calc_window_for_number(df_column, number=0):
    temp = pd.DataFrame()
    temp[""] = df_column
    temp[""] = temp[temp[""] == number]
    temp = temp.dropna()
    return temp.index[0], temp.index[-1]


# calculate windows for all identifier numbers
def calc_windows(df_column):
    output = []
    for n in range(df_column.min().astype(int), df_column.max().astype(int) + 1):
        output.append(__calc_window_for_number(df_column, n))
    return output


# calculate the overlap between two windows
def __compare_two_windows(window1_start, window1_end, window2_start, window2_end, index1, index2, overlap=0.8):
    window1 = pd.DataFrame(index=index1)
    window2 = pd.DataFrame(index=index2)
    output = pd.DataFrame(index=index1 & index2)
    window1[""] = 0
    window2[""] = 0
    window1[window1_start:window1_end] = 0.5
    window2[window2_start:window2_end] = 0.5
    count1 = (window1 == 0.5).values.sum()
    count2 = (window2 == 0.5).values.sum()
    count = max(count1, count2)

    output = window1 + window2
    count_output = (output == 1).values.sum()

    if count_output / count < overlap:
        output[""] = 0
    return output[""]


# calculate the overlap between two dataframes (result is a mask for the data where the overlap is above the cutoff)
def calc_windows_between_two_dataframes(df1_column, df2_column, overlap=0.8):
    df1_windows = calc_windows(df1_column)
    df2_windows = calc_windows(df2_column)

    output = pd.DataFrame(index=df1_column.index & df2_column.index)
    df_out_list = []
    for i in range(len(df1_windows)):
        for j in range(len(df2_windows)):
            df_out_list.append(
                __compare_two_windows(df1_windows[i][0], df1_windows[i][1], df2_windows[j][0], df2_windows[j][1],
                                      df1_column.index, df2_column.index, overlap))
    output = pd.concat(df_out_list)
    output = output.groupby(output.index).max()
    return output
