from scripts.preprocessing import *
import os
import plotly.graph_objects as go
import plotly
import pandas as pd


# use preprocessing.py to import data
def bin_to_dataframe(file):
    data = readBinFile(file)
    dataframe = decompress(data)
    dataframe_equidistant = make_equidistant([dataframe], 25)[0][0]
    return resample_raw_data(dataframe_equidistant, 25)


# import data from specified directory and list of mac addresses and output a dictionary of dataframes
def import_folder(mac_address_list, folder):
    output_dict = {}
    for mac_address in mac_address_list:
        data = []
        for file in os.listdir(folder):
            if file.startswith(mac_address + "_d") & file.endswith(".bin"):
                data.append(bin_to_dataframe(folder + "/" + file))
        if len(data) > 0:
            # output_dict[mac_address] = pd.concat(data)
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
