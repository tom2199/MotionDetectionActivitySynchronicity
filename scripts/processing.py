from scripts.preprocessing import *
import os
import plotly.graph_objects as go


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
            output_dict[mac_address] = data
        else:
            print("No data found for " + mac_address)

    return output_dict


# put all dataframes of a list into a list of traces for plotly
def trace_dataframe_list(dataframe_list, sensor_names):
    output = []
    for dataframe in dataframe_list:
        y_axis = dataframe[sensor_names]
        x_axis = dataframe.index
        for s in sensor_names:
            output.append(go.Scatter(x=x_axis, y=y_axis[s], name=s))
    return output
