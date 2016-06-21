#!/usr/bin/env python
import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import numpy as np

def get_log_parsing_script():
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return dirname + '/parse_log.py'

def get_log_file_suffix():
    return '.log'

def get_chart_type_description_separator():
    return '  vs. '

def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields

def create_field_index():
    train_key = 'Train'
    test_key = 'Test'
    field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' loss':3,
                              train_key + ' learning rate':2},
                   test_key:{'Iters':0, 'Seconds':1, test_key + ' accuracy':3,
                             test_key + ' loss':4}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in xrange(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in xrange(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types()
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description
def get_chart_type_descriptions(chart_types):
    chart_type_descriptions = []
    for ch_type in chart_types:
        chart_type_descriptions.append(get_chart_type_description(ch_type))
    return ' & '.join(chart_type_descriptions)

def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type

def get_data_file(chart_type, path_to_log):
    return os.path.basename(path_to_log) + '.' + get_data_file_type(chart_type).lower()

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field    

def get_field_indecies(x_axis_field, y_axis_field,chart_type):
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index()[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for idx,line in enumerate(f):
            if idx==0:
                continue
            line = line.strip()
            if line[0] != '#':
                fields = line.split(',')
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data
def smooth_data(data,step):
    num_all = np.int(len(data[0]) / step)
    new_data=[[],[]]
    for i in range(num_all):
        sum_data = 0
        count = 0
        for j in range(i*step,min((i+1)*step,len(data[0]))):
            sum_data += data[0][j]
            count+=1
        sum_data /= np.float(count)
        new_data[0].append(sum_data)
        new_data[1].append(data[1][min((i+1)*step,len(data[0]))-1])
    return new_data
def smooth_data2(data,step):
    half_step = np.int(step/2.0)
    len_data = len(data[1])-1
    new_data=[data[0],[]]
    for i in range(len(data[1])):
        #print i
        if i==0:
            sum_data = 0
            for j in range(-half_step,half_step):
                idx = j+i
                idx = min(max(idx,0),len_data)
                sum_data+=data[1][idx]
        else:
            idx2 = max(-half_step+i-1,0)
            idx3 = min(half_step-1+i,len_data)
            sum_data =sum_data-data[1][idx2]+data[1][idx3]
        ave_data = sum_data / step
        #print ave_data,data[1][i]
        new_data[1].append(ave_data)
    return new_data
def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.values())
    idx = random.randint(0, num - 1)
    return markers.values()[idx]

def get_data_label(path_to_log,chart_type):
    post_fix = get_chart_type_description(chart_type).split(get_chart_type_description_separator())[0].replace(' ','_')
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())] + '_' + post_fix
    return label

def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        #loc = 'upper right'
		pass
    return loc

def plot_chart(chart_type, path_to_png, path_to_log_list):
    for path_to_log in path_to_log_list:
        os.system('python %s %s %s' % (get_log_parsing_script(), path_to_log,os.path.dirname(os.path.abspath(path_to_png))))
        for ix,ch_type in enumerate(chart_type):
            data_file = get_data_file(ch_type, path_to_log)
            x_axis_field, y_axis_field = get_field_descriptions(ch_type)
            x, y = get_field_indecies(x_axis_field, y_axis_field,ch_type)
            data = load_data(data_file, x, y)
            #if ix==0:
            #    data = smooth_data2(data,50)
            ## TODO: more systematic color cycle for lines
            color = [random.random(), random.random(), random.random()]
            label = get_data_label(path_to_log,ch_type)
            linewidth = 0.75
            ## If there too many datapoints, do not use marker.
    ##        use_marker = False
            use_marker = True
            if not use_marker:
                plt.plot(data[0], data[1], label = label, color = color,
                         linewidth = linewidth)
            else:
                ok = False
                ## Some markers throw ValueError: Unrecognized marker style
                while not ok:
                    try:
                        marker = random_marker()
                        plt.plot(data[0], data[1], label = label, color = color,
                                 marker = marker, linewidth = linewidth)
                        ok = True
                    except:
                        pass
    legend_loc = get_legend_loc(chart_type[0])
    plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    plt.title(get_chart_type_descriptions(chart_type))
    plt.xlabel(x_axis_field)
    plt.ylabel(y_axis_field)
    #plt.ylim((0,7))
    plt.grid()
    plt.savefig(path_to_png)
    plt.show()

def print_help():
    print """This script mainly serves as the basis of your customizations.
Customization is a must.
You can copy, paste, edit them in whatever way you want.
Be warned that the fields in the training log may change in the future.
You had better check the data files and change the mapping from field name to
 field index in create_field_index before designing your own plots.
Usage:
    ./plot_training_log.py chart_type[0-%s] /where/to/save.png /path/to/first.log ...
Notes:
    1. Supporting multiple logs.
    2. Log file name must end with the lower-cased "%s".
Supported chart types:""" % (len(get_supported_chart_types()) - 1,
                             get_log_file_suffix())
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    for i in xrange(num):
        print '    %d: %s' % (i, supported_chart_types[i])
    exit

def is_valid_chart_type(chart_type):
    flag = True
    for ct in chart_type:
        flag = flag and (ct >= 0 and ct < len(get_supported_chart_types()))

    return flag
  
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_help()
    else:
        chart_type = [int(ct) for ct in sys.argv[1].split(',')]
        if not is_valid_chart_type(chart_type):
            print_help()
        path_to_png = sys.argv[2]
        if not path_to_png.endswith('.png'):
            print 'Path must ends with png' % path_to_png
            exit            
        path_to_logs = sys.argv[3:]
        for path_to_log in path_to_logs:
            if not os.path.exists(path_to_log):
                print 'Path does not exist: %s' % path_to_log
                exit
            if not path_to_log.endswith(get_log_file_suffix()):
                print_help()
        ## plot_chart accpets multiple path_to_logs
        plot_chart(chart_type, path_to_png, path_to_logs)
