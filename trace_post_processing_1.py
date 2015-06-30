#!/usr/bin/python
"""This is a python script version of the iPython notebook - not tested
30/6/2015 DJH"""

import pandas as pd
import os, glob
import numpy as np

daily_input_path = os.path.join('data', 'output', 'd')
monthly_output_path = os.path.join('data', 'output', 'm')
annual_output_path = os.path.join('data', 'output', 'y')
total_output_path = os.path.join('data', 'output', 't')
map_path =  os.path.join('data', 'input', 'maps')

# define function that returns the mean dataframe from a panel of all given dateframes

def mean_results(input_path, year, month, trace_type='td'):
    """Function to return mean results over the month"""
    P = {}
    files = glob.glob(os.path.join(daily_input_path, trace_type + '_' + str(year) + str(month).zfill(2) + '*.csv'))
    for i, f in enumerate(files):
        #print i, f
        P[i] = pd.read_csv(f, index_col = 0)
    return pd.Panel(P).fillna(0.0).mean(0).fillna(0.0)  # take mean over the panel making sure to fill NaNs with zero!

# loop over daily trace results and save monthly mean results to m dir- this works but is very slow!

for trace_type in ['td', 'tu', 'sd', 'su']:
    for YYYY in range(2011,2014):
        for MM in range(1,13):
            output_filename = os.path.join(monthly_output_path, trace_type + '_' + str(YYYY) + str(MM).zfill(2) + '.csv')
            print "Returning mean for " + trace_type + ': ' + str(YYYY) + "/" + str(MM).zfill(2) + ' to ' + output_filename
            mean_results(daily_input_path, YYYY, MM, trace_type=trace_type).to_csv(output_filename)

# combine monthlies into annuals and save to csv in y dir
# define function that returns the mean dataframe from a panel of all given dateframes
def mean_results(input_path, year, trace_type='td'):
    """Function to return mean results over the month"""
    P = {}
    files = glob.glob(os.path.join(input_path, trace_type + '_' + str(year) + '*.csv'))
    for i, f in enumerate(files):
        #print i, f
        P[i] = pd.read_csv(f, index_col = 0)
    return pd.Panel(P).fillna(0.0).mean(0).fillna(0.0)

for trace_type in ['td', 'tu', 'sd', 'su']:
    for YYYY in range(2011,2014):
        output_filename = os.path.join(annual_output_path, trace_type + '_' + str(YYYY) + '.csv')
        print "Returning mean for " + trace_type + ': ' + str(YYYY) + ' to ' + output_filename
        mean_results(monthly_output_path, YYYY, trace_type=trace_type).to_csv(output_filename)

# combine annuals into total for 3 years and save to csv in t dir

def mean_results(input_path, trace_type='td'):
    """Function to return mean results over the month"""
    P = {}
    files = glob.glob(os.path.join(input_path, trace_type + '*.csv'))
    for i, f in enumerate(files):
        #print i, f
        P[i] = pd.read_csv(f, index_col = 0)
    return pd.Panel(P).fillna(0.0).mean(0).fillna(0.0)

for trace_type in ['td', 'tu', 'sd', 'su']:
    output_filename = os.path.join(total_output_path, trace_type + '.csv')
    print "Returning mean for " + trace_type + ': to ' + output_filename
    mean_results(annual_output_path, trace_type=trace_type).to_csv(output_filename)

# load the total trace matrices over the tree years

td = pd.read_csv(os.path.join(total_output_path, 'td.csv'), index_col=0)
tu = pd.read_csv(os.path.join(total_output_path, 'tu.csv'), index_col=0)
sd = pd.read_csv(os.path.join(total_output_path, 'sd.csv'), index_col=0)
su = pd.read_csv(os.path.join(total_output_path, 'su.csv'), index_col=0)

# groupby ELB company for downstream matrices and GIP for generation

# load the GXP to Line company mapping and export to dictionary
elb2gxp = pd.read_csv(os.path.join(map_path, 'elb2gxp.csv'), index_col=0, header=None)[1].to_dict()

# sum downstream GXPs by line company
td.columns = td.columns.map(lambda x: elb2gxp[x[0:7]])
td = td.groupby(level=0, axis=1, sort=False).sum()

# substation.
sd = sd.ix[sd.sum(axis=1) > 0, sd.sum() > 0]  # only if there are flows
sd.columns = sd.columns.map(lambda x: elb2gxp[x[0:7]])
sd = sd.groupby(level=0, axis=1, sort=False).sum()

# Ditto for upstream by generator
tu.columns = tu.columns.map(lambda x: elb2gxp[x[0:7]])
tu = tu.groupby(level=0, axis=1, sort=False).sum()

su = su.ix[su.sum(axis=1) > 0, su.sum() > 0]  # only if there are flows
su.columns = su.columns.map(lambda x: elb2gxp[x[0:7]])
su = su.groupby(level=0, axis=1, sort=False).sum()


sd = sd.groupby(sd.index.map(lambda x: x[0:3]), sort=False).sum()
su = su.groupby(su.index.map(lambda x: x[0:3]), sort=False).sum()

# add HHI columns to each trace output matrix

def hhi(dfrow):
    '''HHI calculator, use with apply etc, eg., df.apply(lambda x: hhi(x),axis=1)'''
    row = [float(ihhi) for ihhi in dfrow]
    return np.sum(((row/np.sum(row))**2))*10000

td['HHI'] = td.apply(lambda x: hhi(x), axis=1)
tu['HHI'] = tu.apply(lambda x: hhi(x), axis=1)
sd['HHI'] = sd.apply(lambda x: hhi(x), axis=1)
su['HHI'] = su.apply(lambda x: hhi(x), axis=1)

# save to csv
td.to_csv(os.path.join(total_output_path, 'td_hhi.csv'))
tu.to_csv(os.path.join(total_output_path, 'tu_hhi.csv'))
sd.to_csv(os.path.join(total_output_path, 'sd_hhi.csv'))
su.to_csv(os.path.join(total_output_path, 'su_hhi.csv'))

