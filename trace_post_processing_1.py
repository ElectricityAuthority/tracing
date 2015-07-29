#!/usr/bin/python
"""This file takes the daily trace data and combines it into monthly, annual and
   total over the time period.  It also removes additional data from the
   substation arrays, with suitable mappings and outputs data n a form for
   further TPM design/analysis."""

import pandas as pd
import os
import glob
import numpy as np

run = 'tpm'

daily_input_path = os.path.join('data', 'output', run, 'd')
monthly_output_path = os.path.join('data', 'output', run,  'm')
annual_output_path = os.path.join('data', 'output', run, 'y')
total_output_path = os.path.join('data', 'output', run, 't')
map_path = os.path.join('data', 'input', 'maps')

# define function that returns the mean dataframe from panel of all dateframes


def mean_results(input_path, year, month, trace_type='td'):
    """Function to return mean results over the month"""
    P = {}
    files = glob.glob(os.path.join(daily_input_path, trace_type + '_' +
                                   str(year) + str(month).zfill(2) + '*.csv'))
    for i, f in enumerate(files):
        P[i] = pd.read_csv(f, index_col=0)
    return pd.Panel(P).fillna(0.0).mean(0).fillna(0.0)  # remember NaNs 2 0!

# loop over daily results, save monthly means to m dir- this is slow...!

for trace_type in ['td', 'tu', 'sd', 'su']:
    for YYYY in range(2011, 2014):
        for MM in range(1, 13):
            output_filename = os.path.join(monthly_output_path, trace_type + '_'
                                           + str(YYYY) + str(MM).zfill(2) + '.csv')
            print "Returning mean for " + trace_type + ': ' + str(YYYY) + "/" \
            + str(MM).zfill(2) + ' to ' + output_filename
            mean_results(daily_input_path, YYYY, MM,
                         trace_type=trace_type).to_csv(output_filename, float_format='%.4f')

# combine monthlies into annuals and save to csv in y dir
# define function that returns the mean dataframe from a panel of all given
# dateframes


def mean_results(input_path, year, trace_type='td'):
    """Function to return mean results over the month"""
    P = {}
    files = glob.glob(os.path.join(input_path, trace_type + '_' + str(year) + '*.csv'))
    for i, f in enumerate(files):
        # print i, f
        P[i] = pd.read_csv(f, index_col=0)
    return pd.Panel(P).fillna(0.0).mean(0).fillna(0.0)

for trace_type in ['td', 'tu', 'sd', 'su']:
    for YYYY in range(2011, 2014):
        output_filename = os.path.join(annual_output_path, trace_type + '_' +
                                       str(YYYY) + '.csv')
        print "Returning mean for " + trace_type + ': ' + str(YYYY) + ' to ' + \
            output_filename
        mean_results(monthly_output_path, YYYY, trace_type=trace_type) \
            .to_csv(output_filename, float_format='%.4f')

# combine annuals into total for 3 years and save to csv in t dir


def mean_results(input_path, trace_type='td'):
    """Function to return mean results over the month"""
    P = {}
    files = glob.glob(os.path.join(input_path, trace_type + '*.csv'))
    for i, f in enumerate(files):
        # print i, f
        P[i] = pd.read_csv(f, index_col=0)
    return pd.Panel(P).fillna(0.0).mean(0).fillna(0.0)

for trace_type in ['td', 'tu', 'sd', 'su']:
    output_filename = os.path.join(total_output_path, trace_type + '.csv')
    print "Returning mean for " + trace_type + ': to ' + output_filename
    mean_results(annual_output_path, trace_type=trace_type).to_csv(output_filename, float_format='%.4f')

# load the total trace matrices over the three years

td = pd.read_csv(os.path.join(total_output_path, 'td.csv'), index_col=0)
tu = pd.read_csv(os.path.join(total_output_path, 'tu.csv'), index_col=0)
sd = pd.read_csv(os.path.join(total_output_path, 'sd.csv'), index_col=0)
su = pd.read_csv(os.path.join(total_output_path, 'su.csv'), index_col=0)

# load the row and column mappings and export to dictionary

nodes_gen = pd.read_csv(os.path.join(map_path, 'nodes_gen.csv'), index_col=0,
                        header=None)[1].to_dict()
nodes_load = pd.read_csv(os.path.join(map_path, 'nodes_load.csv'), index_col=0,
                         header=None)[1].to_dict()
trans_gen = pd.read_csv(os.path.join(map_path, 'trans_gen.csv'), index_col=0,
                        header=None)[1].to_dict()
trans_load = pd.read_csv(os.path.join(map_path, 'trans_load.csv'), index_col=0,
                         header=None)[1].to_dict()


def rollup(df, trans, nodes):
    """rollup output data with above mappings"""
    df.index = df.index.map(lambda x: trans.get(x, None))
    df = df.groupby(level=0, axis=0, sort=False).sum()
    df.columns = df.columns.map(lambda x: nodes.get(x, None))
    return df.groupby(level=0, axis=1, sort=False).sum()

sd = rollup(sd, trans_load, nodes_load)
su = rollup(su, trans_gen, nodes_gen)
td = rollup(td, trans_load, nodes_load)
tu = rollup(tu, trans_gen, nodes_gen)

# su.index = su.index.map(lambda x: trans_gen.get(x, None))
# su = su.groupby(level=0, axis=0, sort=False).sum()
# su.columns = su.columns.map(lambda x: nodes_gen.get(x, None))
# su = su.groupby(level=0, axis=1, sort=False).sum()

# td.index = td.index.map(lambda x: trans_load.get(x, None))
# td = td.groupby(level=0, axis=0, sort=False).sum()
# td.columns = td.columns.map(lambda x: nodes_load.get(x, None))
# td = td.groupby(level=0, axis=1, sort=False).sum()

# tu.index = tu.index.map(lambda x: trans_gen.get(x, None))
# tu = tu.groupby(level=0, axis=0, sort=False).sum()
# tu.columns = tu.columns.map(lambda x: nodes_gen.get(x, None))
# tu = tu.groupby(level=0, axis=1, sort=False).sum()

# save to csv
td.to_csv(os.path.join(total_output_path, 'td_spreadsheet.csv'), float_format='%.4f')
tu.to_csv(os.path.join(total_output_path, 'tu_spreadsheet.csv'), float_format='%.4f')
sd.to_csv(os.path.join(total_output_path, 'sd_spreadsheet.csv'), float_format='%.4f')
su.to_csv(os.path.join(total_output_path, 'su_spreadsheet.csv'), float_format='%.4f')

# group by ELB company for downstream matrices and GIP for generation
# load the GXP to Line company mapping and export to dictionary
elb2gxp = pd.read_csv(os.path.join(map_path, 'elb2gxp.csv'), index_col=0,
                      header=None)[1].to_dict()

# sum downstream GXPs by line company
td.columns = td.columns.map(lambda x: elb2gxp[x[0:7]])
td = td.groupby(level=0, axis=1, sort=False).sum()

# substation
sd = sd.ix[sd.sum(axis=1) > 0, sd.sum() > 0]  # only if there are flows
sd.columns = sd.columns.map(lambda x: elb2gxp[x[0:7]])
sd = sd.groupby(level=0, axis=1, sort=False).sum()

# ditto for upstream by generator
tu.columns = tu.columns.map(lambda x: elb2gxp[x[0:7]])
tu = tu.groupby(level=0, axis=1, sort=False).sum()

su = su.ix[su.sum(axis=1) > 0, su.sum() > 0]  # only if there are flows
su.columns = su.columns.map(lambda x: elb2gxp[x[0:7]])
su = su.groupby(level=0, axis=1, sort=False).sum()


sd = sd.groupby(sd.index.map(lambda x: x[0:3]), sort=False).sum()
su = su.groupby(su.index.map(lambda x: x[0:3]), sort=False).sum()

# add HHI columns to each trace output matrix


def hhi(dfrow):
    '''HHI calculator, use with apply etc, eg.,
       df.apply(lambda x: hhi(x),axis=1)'''
    row = [float(ihhi) for ihhi in dfrow]
    return np.sum(((row/np.sum(row))**2))*10000

td['HHI'] = td.apply(lambda x: hhi(x), axis=1)
tu['HHI'] = tu.apply(lambda x: hhi(x), axis=1)
sd['HHI'] = sd.apply(lambda x: hhi(x), axis=1)
su['HHI'] = su.apply(lambda x: hhi(x), axis=1)

# save to csv
td.to_csv(os.path.join(total_output_path, 'td_hhi.csv'), float_format='%.4f')
tu.to_csv(os.path.join(total_output_path, 'tu_hhi.csv'), float_format='%.4f')
sd.to_csv(os.path.join(total_output_path, 'sd_hhi.csv'), float_format='%.4f')
su.to_csv(os.path.join(total_output_path, 'su_hhi.csv'), float_format='%.4f')

# reload the total trace matrices over the three years
# td = pd.read_csv(os.path.join(total_output_path, 'td.csv'), index_col=0)
# tu = pd.read_csv(os.path.join(total_output_path, 'tu.csv'), index_col=0)
# sd = pd.read_csv(os.path.join(total_output_path, 'sd.csv'), index_col=0)
# su = pd.read_csv(os.path.join(total_output_path, 'su.csv'), index_col=0)

