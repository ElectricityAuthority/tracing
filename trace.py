#!/usr/bin/python
"""TODO list:
  - add command line inputs
  - and options for outputs i.e., ELB/node etc.
  - general code readability improvements etc"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging as l
from psutil import phymem_usage
import os
import traceback


# Setup some console logging
formatter = l.Formatter('|%(asctime)-6s|%(message)s|', '%Y-%m-%d %H:%M:%S')
conlog = l.StreamHandler()
conlog.setLevel(l.INFO)
conlog.setFormatter(formatter)
l.getLogger('').addHandler(conlog)
logger = l.getLogger('TRACE')
logger.setLevel(l.INFO)


def load_vSPD_data(vSPD_b, vSPD_n, mappings=True):
    """Function that loads vSPD data"""
    def bmmap(x):
        """Extra nodes inserted by Concept"""
        bm = {'BEN0162': 6000,
              'BEN0163': 6001,
              'TKB2202': 6002,
              'GOR2201': 6003,
              'TKA0112': 6004,
              'WRU2201': 6005,
              'HOB2201': 6006}
        if x in bm.keys():
            return bm[x]
        else:
            return int(x)

    # Load annual Node/Bus data
    n = pd.read_csv(vSPD_n, header=None, index_col=0, parse_dates=True,
                    names=['tp', 'node', 'bus', 'allofact', 'GENERATION',
                           'LOAD', 'bidMW'])
    n.bus = n.bus.map(lambda x: bmmap(x))
    # Load branch flow data
    b = pd.read_csv(vSPD_b, header=None, index_col=0,
                    names=['tp', 'branch', 'FROM_MW', 'DynamicLoss (MW)',
                           'FixedLoss (MW)', 'FROM_ID_BUS', 'TO_ID_BUS'],
                    parse_dates=True)
    b.FROM_ID_BUS = b.FROM_ID_BUS.map(lambda x: bmmap(x))
    b.TO_ID_BUS = b.TO_ID_BUS.map(lambda x: bmmap(x))
    if mappings:
        # Get some mappings
        nmap = n.ix[:, ['node', 'bus']].reset_index(drop=True)\
                .drop_duplicates().set_index(['bus']).sort_index().node
        nmap2 = nmap.groupby(level=0).first()
        brmap = b.ix[:, ['branch', 'FROM_ID_BUS', 'TO_ID_BUS']]\
                .reset_index(drop=True).drop_duplicates()\
                .set_index(['FROM_ID_BUS', 'TO_ID_BUS'])['branch']
    # Further munging
    n = n.set_index(['tp', 'node', 'bus'], append=True)
    n['LOAD'] = n.allofact * (n.LOAD + n.bidMW)
    n['GENERATION'] = n.allofact * n.GENERATION
    n = n.drop(['allofact', 'bidMW'], axis=1)
    b['TO_MW'] = b['FROM_MW'] - b['DynamicLoss (MW)']
    b = b.set_index(['tp', 'branch', 'FROM_ID_BUS', 'TO_ID_BUS'], append=True)
    b = b.ix[:, ['FROM_MW', 'TO_MW']]

    if mappings:
        return n, b, nmap, nmap2, brmap
    else:
        return n, b


def A(b, n, downstream=True):
    """Given branch flows and load/generation build the A matrix and solve."""
    b = b.ix[:, ['FROM_MW', 'TO_MW']]  # grab the columns of interest
    allbus = list(set(b.index.levels[0]) | set(b.index.levels[1]))
    totbus = len(allbus)
    pg = n.GENERATION.ix[allbus].fillna(0.0).groupby(level=0).sum()
    pl = n.LOAD.ix[allbus].fillna(0.0).groupby(level=0).sum()
    A = pd.DataFrame(np.identity(totbus), index=allbus,
                     columns=allbus).fillna(0.0)
    if downstream:
        """Build Nett downstream Ad with inflows"""
        b1 = b.ix[b.FROM_MW < 0].FROM_MW  # FROM_MW neg, into bus
        b2 = b.swaplevel(0, 1).sort()  # swap levels and sort
        b2 = -b2.ix[b2.TO_MW > 0].TO_MW   # TO_MW pos, into bus
    else:
        """Build Gross upstream Au with outflows"""
        b1 = b.ix[b.TO_MW < 0].TO_MW
        b2 = b.swaplevel(0, 1).sort()
        b2 = -b2.ix[b2.FROM_MW > 0].FROM_MW

    b2.index.names = ['FROM_ID_BUS', 'TO_ID_BUS']  # rename to match flow
    cji = b1.append(b2).sortlevel()  # append and sort
    cji = cji.groupby(level=[0, 1]).sum()  # sum parallel branches

    # calculate Nodal through-flows using inflows
    inflow1 = -b.ix[b.FROM_MW < 0].groupby(level=0).sum().FROM_MW  # out
    inflow2 = b.ix[b.TO_MW > 0].groupby(level=1).sum().TO_MW
    inflow = pd.DataFrame({'inflow1': inflow1, 'inflow2': inflow2})\
               .fillna(0.0).sum(axis=1)
    # calculate Nodal through-flows using outflows
    outflow1 = b.ix[b.FROM_MW > 0].groupby(level=0).sum().FROM_MW
    outflow2 = -b.ix[b.TO_MW < 0].groupby(level=1).sum().TO_MW
    outflow = pd.DataFrame({'outflow1': outflow1,
                            'outflow2': outflow2}).fillna(0.0).sum(axis=1)
    Pi = pd.DataFrame({'inflow': inflow, 'outflow': outflow}).fillna(0.0)
    PI = pd.DataFrame({'Pi+pg': Pi.inflow.add(pg,
                                              fill_value=0).fillna(0.0),
                       'Pi+pl': Pi.outflow.add(pl,
                                               fill_value=0).fillna(0.0)})
    if downstream:
        """Build Nett downstream Ad with inflows"""
        for i, r in pd.DataFrame({'cji': cji}).iterrows():
            v = r.values/(PI['Pi+pl'].ix[i[0]])
            A.ix[i[1], i[0]] = v
    else:
        """Build Gross upstream Au with outflows"""
        for i, r in pd.DataFrame({'cji': cji}).iterrows():
            A.ix[i[0], i[1]] = r.values/(PI['Pi+pg'].ix[i[1]])

    A = A.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    iA = np.linalg.inv(A)  # invert
    Pi = PI['Pi+pg'].ix[allbus]
    if downstream:  # calculate the nett generation
        pgn = iA.dot(np.array(pl.values))*np.array(pg.values) \
            * np.array(((1 / Pi).fillna(0.0)).values)
        pg = pd.Series(pgn, index=allbus).fillna(0.0)
    else:  # calculate the gross demand
        plg = iA.dot(np.array(pg.values)) * np.array(pl.values) \
            * np.array(((1 / Pi).fillna(0.0)).values)
        pl = pd.Series(plg, index=allbus).fillna(0.0)
    return A, iA, pg, pl, b2, cji, Pi


def topo(iA, pi, bd, plg, downstream=True):
    """Calculate the topological distribution matrices, this needs a bit of
       a tidy up"""
    # common to both gross-up and nett-down
    allbus = list(set(bd.index.levels[0]) | set(bd.index.levels[1]))
    totbus = len(allbus)
    bdd = bd.groupby(level=[0, 1]).sum().dropna()
    bpos = bdd.TO_MW >= 0  # Masks that depend on flow direction
    bneg = bdd.TO_MW < 0
    bposx = np.array([bpos.values, ] * totbus).transpose()
    bnegx = np.array([bneg.values, ] * totbus).transpose()
    ibus = bdd.reset_index().TO_ID_BUS.values
    jbus = bdd.reset_index().FROM_ID_BUS.values
    pii = np.array([pi.ix[ibus].values, ] * totbus).transpose()
    pij = np.array([pi.ix[jbus].values, ] * totbus).transpose()
    b_in = bdd.TO_MW.values
    b_out = bdd.FROM_MW.values
    b_inx = np.array([b_in, ]*totbus).transpose()
    b_outx = np.array([b_out, ]*totbus).transpose()
    if downstream:  # Calculate nett branch flows for downstream to demand
        iAd_df = pd.DataFrame(iA, index=allbus, columns=allbus)
        i_Ad_ibus = iAd_df.ix[ibus, :]
        i_Ad_jbus = iAd_df.ix[jbus, :]
        Pdd = [plg.values, ]*len(b_in)
        DDilk1 = np.abs(b_inx) * i_Ad_jbus.values * Pdd * (1 / pij)
        DDilk2 = np.abs(b_outx) * i_Ad_ibus.values * Pdd * (1 / pii)
        dfd = DDilk1*bnegx + DDilk2*bposx
        dfd = pd.DataFrame(dfd, index=bdd.index, columns=allbus).fillna(0.0)
        idx = list(set(dfd.columns) & set(nmap.index))  # filter columns
        df = dfd[idx]
    else:  # Calculate gross branch flows for upstream to generators
        iAu_df = pd.DataFrame(iA, index=allbus, columns=allbus)
        i_Au_ibus = iAu_df.ix[ibus, :]
        i_Au_jbus = iAu_df.ix[jbus, :]
        Pgd = [plg.values, ] * len(b_in)
        DGilk1 = np.abs(b_inx) * i_Au_ibus.values * Pgd * (1 / pii)
        DGilk2 = np.abs(b_outx) * i_Au_jbus.values * Pgd * (1 / pij)
        dfg = DGilk1 * bposx + DGilk2 * bnegx
        dfg = pd.DataFrame(dfg, index=bdd.index, columns=allbus).fillna(0.0)
        idx = list(set(dfg.columns) & set(nmap.index))  # filter columns
        df = dfg[idx]
    return df


def bustocomp(df, nmap, brmap, NPmap, node=False):
    """Given bus level data, group up to either ELB level,
        node=False, or node level, node=True and map to branch names"""
    # Reindex no node labels
    df.index = df.index.map(lambda x: brmap[x])
    df.columns = df.columns.map(lambda x: nmap[x])
    # select rows and columns that sum to greater than 0
    df = df.ix[df.sum(axis=1) > 0, df.sum() > 0]
    # Sum columns to the node label level
    df = df.groupby(level=0, axis=1, sort=False).sum()

    def seriestolist(x):
        """We get pd.Series objects when mapping to branch name level,
            objects from series to array types"""
        if isinstance(x, type(pd.Series())):
            idx = x.values
            return ', '.join(str(e) for e in idx)
        else:
            return x
    df.index = df.index.map(lambda x: seriestolist(x))
    if node:
        return df
    else:  # Sum by ELB
        df.columns = df.columns.map(lambda x: NPmap[x[0:7]])
        df = df.groupby(level=0, axis=1, sort=False).sum()
        return df


def trans_use(b, n, nmap, brmap, NPmap, downstream=True):
    """Subroutine to calculate transmission usage matrix"""
    #if downstream:  # Net downstream pg is nett of losses, pl is actual
        #Ad, iAd, pg, pl, bdd, cjid, Pi = A(b, n, downstream=downstream)
        #df = topo(iAd, Pi, b, pl, downstream=True)
        #df1 = bustocomp(df.copy(), nmap, brmap, NPmap, node=True)  # Node level,
        #df2 = bustocomp(df.copy(), nmap, brmap, NPmap)  # ELB level
    #else:  # Gross upstream pg is actual, pl grossed with losses
        #Au, iAu, pg, pl, bdu, cjiu, Pi = A(b, n, downstream=downstream)
        #df = topo(iAu, Pi, b, pg, downstream=False)
        #df1 = bustocomp(df.copy(), nmap, brmap, NPmap, node=True)  # Node level,
        #df2 = bustocomp(df.copy(), nmap, brmap, NPmap)  # ELB level
    Ac, iAc, pg, pl, bd, cji, Pi = A(b, n, downstream=downstream)
    df = topo(iAc, Pi, b, pl, downstream=downstream)
    df1 = bustocomp(df.copy(), nmap, brmap, NPmap, node=True)  # Node level,
    df2 = bustocomp(df.copy(), nmap, brmap, NPmap)  # ELB level
    #return df, df1, df2, pl, pg
    return df, df1, df2, pl, pg


def sub_usage(df, pl, pg, nmap, NPmap):
    """Calculate substation usage matrix.  Groups to node level, then substation
       level.  MW values are summed at substation level so represent total
       through flow, summed through all buses that comprise a substation.  As a
       substation consists of multiple buses, the resulting flows are not very
       representative or useful for anything other than calculating usage.
       There are a number of other methods that could be used, i.e, select
       the highest voltage bus."""

    def submapping(NPmap):
        """Determine the ELB/substation mapping given node/ELB mapping """
        submap = pd.DataFrame({'comp': NPmap})
        submap.index = submap.index.map(lambda x: x[0:3])
        submap = submap.reset_index().drop_duplicates()\
                       .set_index('index').comp.to_dict()
        submapex = {'RPO': 'GENE'}
        submap = dict(submap.items() + submapex.items())
        return submap

    def bus_trace_usage(df, pl, pg, nmap):
        """Determine bus usage using traced MW from branch dfd"""
        bus0 = df.groupby(level=0).sum()/2.0
        bus1 = df.groupby(level=1).sum()/2.0
        bus = bus0.append(bus1).groupby(level=0).sum()
        idx = np.array(list(set(bus.columns) | set(bus.index)))
        Lx = pd.DataFrame(np.identity(len(idx)), index=idx, columns=idx)
        pl = pl.ix[idx]
        pg = pg.ix[idx]
        Lxx = Lx * pl/2.0 + Lx * pg/2.0
        bus_usage = (bus.ix[nmap.index, nmap.index] +
                     Lxx.ix[nmap.index, nmap.index]).fillna(0.0)
        return bus_usage

    def bus2node(df, nmap, NPmap):
        """Given bus level data, group up to node level"""
        df.index = df.index.map(lambda x: nmap[x])
        df.columns = df.columns.map(lambda x: nmap[x])
        return df

    def node2sub(df, NPmap, submap):
        """Given node level data, group up to substation level"""
        df = df.ix[df.sum(axis=1) > 0, df.sum() > 0]
        df = df.groupby(df.columns.map(lambda x: NPmap[x[0:7]]),
                        axis=1, sort=False).sum()
        df = df.groupby(df.index.map(lambda x: x[0:3]), sort=False).sum()
        return df

    b_usage = bus_trace_usage(df, pl, pg, nmap)
    n_usage = bus2node(b_usage, nmap, NPmap)
    s_usage = node2sub(n_usage, NPmap, submapping(NPmap))

    return n_usage, s_usage


###############################################################################
# Start TRACE
#
# Loop through monthly branch and node files, we have to do this because we can
# not fit all data (3 years worth) into memory all at once.
# months.  Note: used a cool gawk command line script to split to monthlies...
#
# For each monthly data set, we loop over trading periods.  Note: it would be
# much better suited (cleaner and faster) to use a DataFrame .groupby and .apply
# methods for this... future todo... for now we loop...
#
# For each TP;
#   - perform UP stream trace for generation usage of transmission assets
#   - perform DOWN stream trace for load/demand usage of transmission assets.
#
# To do this we, initiate collection dictionaries then loop over the input data.
#
# D J Hume, Dec 2014.
#
###############################################################################

# Setup paths
path = os.getcwd()
inpath = os.path.join(path, 'data', 'input', 'vSPDout')
mappath = os.path.join(path, 'data', 'input', 'maps')
outpath = os.path.join(path, 'data', 'output')

# Load data mappings

NPmap = pd.read_csv(os.path.join(mappath, 'elb2gxp.csv'), index_col=0,
                    header=None)[1].to_dict()
brmap = pd.read_csv(os.path.join(mappath, 'brmap.csv'), index_col=[0, 1],
                    header=None)[2]
nmap = pd.read_csv(os.path.join(mappath, 'busnode.csv'), index_col=0,
                   header=None)[1]
nmap2 = pd.read_csv(os.path.join(mappath, 'busnode2.csv'), index_col=0,
                    header=None)[1]

logger.info(20*'*')
logger.info("Start tracing routine")
logger.info(20*'*')
fc = {}  # failed counter
test_limit_min = datetime(2011, 1, 1)
test_limit_max = datetime(2013, 12, 31)
for y in [2011, 2012, 2013]:
    if (y <= test_limit_max.year) & (y >= test_limit_min.year):
        for m in range(1, 13):  # load monthly data
            if (m <= test_limit_max.month) & (m >= test_limit_min.month):
                td = {}  # downstream transmission usage
                sd = {}  # downstream substation usage
                tu = {}  # upstream transmission usage
                su = {}  # upstream substation usage
                ym = str(y) + str(m).zfill(2) + '.csv'
                vSPD_b = os.path.join(inpath, 'b_' + ym)
                vSPD_n = os.path.join(inpath, 'n_' + ym)
                info_text = 'INPUT: b_' + ym + ', n_' + ym
                logger.info(info_text)
                n, b = load_vSPD_data(vSPD_b, vSPD_n, mappings=False)
                for day in n.index.levels[0]:
                    """Possible update. Use groupby/apply on pd.Dataframe."""
                    ymd = str(y) + str(m).zfill(2) + str(day.day).zfill(2) + '.csv'
                    if (day <= test_limit_max) & (day >= test_limit_min):
                        for tp in n.index.levels[1]:
                            try:
                                info_text = "TRACE: " + str(day.date()) + "|TP " + \
                                            str(int(tp)).zfill(2) + "|Mem=" + \
                                            str(phymem_usage()[2]) + "%"
                                logger.info(info_text)
                                # get TP level data
                                n2 = n.xs(day, level=0).xs(tp, level=0)\
                                    .reset_index('node', drop=True)
                                b2 = b.xs(day, level=0).xs(tp, level=0)\
                                    .reset_index('branch', drop=True)
                                # Perform downstream trace
                                dfd, dfd1, dfd2, pl, pg = trans_use(b2, n2, nmap2,
                                                                    brmap, NPmap,
                                                                    downstream=True)
                                dfds, dfds2 = sub_usage(dfd, pl, pg, nmap2, NPmap)
                                td[(day, str(tp))] = dfd1
                                sd[(day, str(tp))] = dfds
                                # Perform upstream trace
                                dfu, dfu1, dfu2, pl, pg = trans_use(b2, n2, nmap2,
                                                                    brmap, NPmap,
                                                                    downstream=False)
                                dfus, dfus2 = sub_usage(dfu, pl, pg, nmap2, NPmap)
                                tu[(day, str(tp))] = dfu1
                                su[(day, str(tp))] = dfus
                                fc[(day, str(tp))] = 0
                            except Exception:
                                logger.error("***FAILED*** for " + str(day) +
                                            " trading period " + str(int(tp)) +
                                            "***")
                                logger.error(traceback.print_exc())
                                fc[(day, str(tp))] = 1
                                pass
                # average monthly output filenames (as csv)
                tuc = os.path.join(outpath, 'tu_' + ym)
                suc = os.path.join(outpath, 'su_' + ym)
                tdc = os.path.join(outpath, 'td_' + ym)
                sdc = os.path.join(outpath, 'sd_' + ym)
                # TP level data monthly output filenames (pickles)
                tup = os.path.join(outpath, 'tp', 'tu_' + ym[:6] + '.pickle')
                sup = os.path.join(outpath, 'tp', 'su_' + ym[:6] + '.pickle')
                tdp = os.path.join(outpath, 'tp', 'td_' + ym[:6] + '.pickle')
                sdp = os.path.join(outpath, 'tp', 'sd_' + ym[:6] + '.pickle')
                # panelize, fillna
                TU = pd.Panel(tu).fillna(0.0)
                SU = pd.Panel(su).fillna(0.0)
                TD = pd.Panel(td).fillna(0.0)
                SD = pd.Panel(sd).fillna(0.0)
                # output data files
                TU.to_pickle(tup)
                SU.to_pickle(sup)
                TD.to_pickle(tdp)
                SD.to_pickle(sdp)
                TU.mean(0).to_csv(tuc, float_format='%.4f')
                SU.mean(0).to_csv(suc, float_format='%.4f')
                TD.mean(0).to_csv(tdc, float_format='%.4f')
                SD.mean(0).to_csv(sdc, float_format='%.4f')
                # log
                logger.info(21*'=')
                logger.info("|OUTPUT: " + tup + '|')
                logger.info("|OUTPUT: " + sup + '|')
                logger.info("|OUTPUT: " + tdp + '|')
                logger.info("|OUTPUT: " + sdp + '|')
                logger.info("|OUTPUT: " + tuc + '|')
                logger.info("|OUTPUT: " + suc + '|')
                logger.info("|OUTPUT: " + tdc + '|')
                logger.info("|OUTPUT: " + sdc + '|')

fc = pd.Series(fc).to_csv(os.path.join(outpath, 'fc.csv'))
