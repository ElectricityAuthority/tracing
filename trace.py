#!/usr/bin/python

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import logging as l
# from psutil import phymem_usage
import os
import traceback
import pdb

# setup command line option and argument parsing
parser = argparse.ArgumentParser(description='Flow tracing routine')
parser.add_argument('-type', '--type', choices=['tpm', 'vspd',
                                                'testA', 'testB'],
                    action='store', dest='t', default='tpm',
                    help="""trace type (default = tpm)
                    vSPD output GDX trace = vspd,
                    testA 3 bus test system = testA
                    testB 5 bus Bialek test system = testB""")
parser.add_argument('--tp', action='store_true', dest='tp',
                    help="""trace output at trading period level""")
parser.add_argument('-s', '--start', action='store', dest='s',
                    default="2011-01-01",
                    help="""trace start time (default = 2011-01-01)""")
parser.add_argument('-e', '--end', action='store', dest='e',
                    default="2013-12-31",
                    help="""trace end time (default = 2013-12-31)""")
parser.set_defaults(tp=False)

p = parser.parse_args()

# Setup some console logging
formatter = l.Formatter('|%(asctime)-6s|%(message)s|', '%Y-%m-%d %H:%M:%S')
conlog = l.StreamHandler()
conlog.setLevel(l.INFO)
conlog.setFormatter(formatter)
l.getLogger('').addHandler(conlog)
logger = l.getLogger('TRACE')
logger.setLevel(l.INFO)


class trace():
    """This is the trace class.
    """

    def __init__(self, t):
        # setup paths and create output directory structure if required.
        self.run = p.t
        self.path = os.getcwd()
        self.outpath = os.path.join(self.path, 'data', 'output', self.run)
        self.start = datetime(int(p.s.split('-')[0]),
                              int(p.s.split('-')[1]),
                              int(p.s.split('-')[2]))  # start time
        self.end = datetime(int(p.e.split('-')[0]),
                            int(p.e.split('-')[1]),
                            int(p.e.split('-')[2]))  # end time
        if self.run == 'tpm':
            self.inpath = os.path.join(self.path, 'data', 'input', self.run)
            self.mappath = os.path.join(self.path, 'data', 'input', 'maps')
            # Load data mappings only once for the tpm trace
            self.brmap = pd.read_csv(os.path.join(self.mappath, 'brmap.csv'),
                                     index_col=[0, 1], header=None)[2]
            self.nmap = pd.read_csv(os.path.join(self.mappath, 'busnode.csv'),
                                    index_col=0, header=None)[1]
            # print self.brmap.head()
            # print self.nmap.head()
        if self.run == 'vspd':
            self.ymd = p.s.replace('-', '')  # get day in YYYYMMDD format
            self.inpath = os.path.join(self.path, 'data', 'input', 'vspd',
                                       'vspd', 'Output', self.ymd)
            self.mappath = os.path.join(self.path, 'data', 'input', 'vspd',
                                        'busnodemap', 'mappings')
            self.vspd_brch = os.path.join(self.inpath,
                                          self.ymd + '_BranchResults_TP.csv')
            self.vspd_node = os.path.join(self.inpath,
                                          self.ymd + '_NodeResults_TP.csv')
            self.vspd_busmap = os.path.join(self.mappath,
                                            self.ymd + '.csv')
            self.brch = pd.read_csv(self.vspd_brch,
                                      index_col=0, parse_dates=True)
            self.node = pd.read_csv(self.vspd_node,
                                      index_col=[0, 1], parse_dates=True)
            self.busmap = pd.read_csv(self.vspd_busmap,
                                      index_col=[0, 1, 2], parse_dates=True)

        if (self.run == 'testA') or (self.run == 'testB'):
            self.inpath = os.path.join(self.path, 'data', 'input', self.run,
                                       'input', 'vSPDout')
            self.mappath = os.path.join(self.path, 'data', 'input', self.run,
                                        'input', 'maps')
            # Load data mappings only once for the tpm trace
            self.brmap = pd.read_csv(os.path.join(self.mappath, 'brmap.csv'),
                                     index_col=[0, 1], header=None)[2]
            self.nmap = pd.read_csv(os.path.join(self.mappath, 'busnode.csv'),
                                    index_col=0, header=None)[1]

        self.fc = {}  # failed counter
        self.tp = p.tp  # Trading Period level output

    # def check_args(self):
        # """make sure arguments work with existing code base"""
        # if self.run == "tpm":  # make sure start and end are a month apart
        # if self.end

    def create_output_dir(self):
        if self.run == 'tpm':
            paths = ['tp', 'd', 'm', 'y', 't']
        if self.run == 'vspd':
            paths = ['tp', 'd', 'm', 'y', 't', 'fc']
        else:
            paths = ['d']

        if not os.path.exists(self.outpath):
            for p in paths:
                os.makedirs(os.path.join(self.outpath, p))

    def load_daily_vSPD_data(self, pre_msp=False):
        """function that loads vspd data.  on 21 july, 2009 the input data
        changed and we need to handle bus mappings for this"""

        def swaperator(x):
            """If load is neg put in gen column and vice-versa - slow!"""
            if x.LOAD < 0.0:
                x["GENERATION"] = -x["LOAD"]
                x["LOAD"] = 0.0
            if x["GENERATION"] < 0.0:
                x["LOAD"] = -x["GENERATION"]
                x["GENERATION"] = 0.0
            return x
        busmap = self.busmap
        n = self.node
        b = self.brch
        busmap['Total_Generation'] = busmap.index.map(lambda x: n['Generation (MW)'].ix[(x[0], x[1]), :])
        busmap['Total_Load'] = busmap.index.map(lambda x: n['Load (MW)'].ix[(x[0], x[1]), :])
        busmap['GENERATION'] = busmap.Allo * busmap.Total_Generation
        busmap['LOAD'] = busmap.Allo * busmap.Total_Load
        n = busmap.ix[:, ['GENERATION', 'LOAD']].swaplevel(1, 2).sort_index()
        n = n.apply(lambda x: swaperator(x), axis=1)
        b.rename(columns=dict(zip(b.columns[:4], ['branch', 'FROM_ID_BUS',
                                                  'TO_ID_BUS', 'FROM_MW', ])),
                 inplace=True)
        # print b.head()
        # add/subtract dynamic loss from vSPD output
        # flow>0 flow "from bus"->"to bus": "from bus"=flow; "to bus"=flow-loss
        # flow<0 flow "to bus"->"from bus": "to bus"=flow; "from bus"=flow+loss
        # pdb.set_trace()
        bpos = b.copy().ix[b.FROM_MW >= 0.0]
        bneg = b.copy().ix[b.FROM_MW < 0.0]
        bpos['TO_MW'] = bpos['FROM_MW'] - bpos['DynamicLoss (MW)']
        bneg['TO_MW'] = bneg['FROM_MW']
        bneg['FROM_MW'] = bneg['FROM_MW'] + bneg['DynamicLoss (MW)']
        b = bneg.append(bpos)
        b = b.set_index(['branch', 'FROM_ID_BUS', 'TO_ID_BUS'],
                        append=True)
        b = b.sort_index().ix[:, ['FROM_MW', 'TO_MW']]
        # print n.head()
        # print b.head()
        return n, b


        # def make_int(x):
            # try:
                # return int(x)
            # except:
                # try:
                    # if 'hay' in x:
                        # return int(x.replace('hay', '819'))
                    # if 'ben' in x:
                        # return int(x.replace('ben', '259'))
                # except:
                    # print str(x)
                    # return x
        # # load bus/node mappings
        # if not pre_msp:
            # self.busmap.bus = self.busmap.bus.map(lambda x: make_int(x))
            # self.busmap.set_index('bus', append=true, inplace=true)
        # if pre_msp:  # set integer bus mapping on nodes
            # buses = set(self.busmap.node.values) | set(self.busmap.bus.values)
            # self.busmap = pd.series(list(buses)).reset_index()\
                # .set_index(0)['index'].to_dict()

        # b = pd.read_csv(self.vspd_brch, index_col=0, parse_dates=true)
        # b.rename(columns=dict(zip(b.columns[:4], ['branch', 'from_id_bus',
                                                  # 'to_id_bus', 'from_mw', ])),
                 # inplace=true)
        # # b['to_mw'] = b['from_mw'] - b['dynamicloss (mw)']
        # if not pre_msp:
            # b.from_id_bus = b.from_id_bus.map(lambda x: make_int(x))
            # b.to_id_bus = b.to_id_bus.map(lambda x: make_int(x))
        # if pre_msp:
            # b.from_id_bus = b.from_id_bus.map(lambda x: self.busmap[x])
            # b.to_id_bus = b.to_id_bus.map(lambda x: self.busmap[x])

        # # b = b.set_index(['branch', 'from_id_bus', 'to_id_bus'], append=true)
        # # b = b.ix[:, ['from_mw', 'to_mw']]

        # n = pd.read_csv(self.vspd_node, index_col=0, parse_dates=true)
        # n.rename(columns=dict(zip(n.columns[:3],
                                  # ['bus', 'generation', 'load'])), inplace=true)
        # n = n.ix[(n['generation'] != 0.0) | (n['load'] != 0.0)]
        # if not pre_msp:
            # n.bus = n.bus.map(lambda x: make_int(x))
        # if pre_msp:
            # n['node'] = n.bus
            # n.bus = n.bus.map(lambda x: self.busmap[x])

        # n.set_index('bus', append=true, inplace=true)

        # def add_node(x):
            # """complicated substitution - nodename to bus, but some buses have no
            # nodes?"""
            # try:
                # return self.busmap.loc[x, 'node'].values[-1]
            # except:
                # return np.nan
                # pass
        # if not pre_msp:
            # n['node'] = n.index.map(lambda x: add_node(x))
        # n.set_index('node', append=true, inplace=true)
        # n = n.ix[:, ['generation', 'load']].swaplevel(1, 2).sort_index()

        # def swaperator(x):
            # """if load is neg put in gen column and vice-versa - slow!"""
            # if x.load < 0.0:
                # x["generation"] = -x["load"]
                # x["load"] = 0.0
            # if x["generation"] < 0.0:
                # x["load"] = -x["generation"]
                # x["generation"] = 0.0
            # return x
        # n = n.apply(lambda x: swaperator(x), axis=1)
        # # add/subtract dynamic loss from vspd output
        # # flow>0 flow "from bus"->"to bus": "from bus"=flow; "to bus"=flow-loss
        # # flow<0 flow "to bus"->"from bus": "to bus"=flow; "from bus"=flow+loss
        # # pdb.set_trace()
        # bpos = b.copy().ix[b.from_mw >= 0.0]
        # bneg = b.copy().ix[b.from_mw < 0.0]
        # bpos['to_mw'] = bpos['from_mw'] - bpos['dynamicloss (mw)']
        # bneg['to_mw'] = bneg['from_mw']
        # bneg['from_mw'] = bneg['from_mw'] + bneg['dynamicloss (mw)']
        # b = bneg.append(bpos)
        # # print b.tail()
        # b = b.set_index(['branch', 'from_id_bus', 'to_id_bus'],
                        # append=true)
        # b = b.sort_index().ix[:, ['from_mw', 'to_mw']]
        # # get some mappings
        # nmap = n.reset_index(drop=false).ix[:, ['node', 'bus']]\
            # .drop_duplicates().set_index(['bus']).sort_index().node
        # nmap = nmap.groupby(level=0).first()
        # brmap = b.reset_index(drop=false).ix[:, ['branch', 'from_id_bus',
                                                 # 'to_id_bus']]\
            # .reset_index(drop=true).drop_duplicates()\
            # .set_index(['from_id_bus', 'to_id_bus'])['branch']
        # self.nmap = nmap
        # self.brmap = brmap
        # print n.head()
        # print b.head()
        # return n, b

    def load_tpm_vSPD_data(self, vSPD_b, vSPD_n):
        """Function that loads vSPD data"""
        def bmmap(x):
            """Extra nodes inserted by Concept"""
            bm = {'BEN0162': 6000, 'BEN0163': 6001, 'TKB2202': 6002,
                  'GOR2201': 6003, 'TKA0112': 6004, 'WRU2201': 6005,
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
        n = n.set_index(['tp', 'node', 'bus'], append=True)
        n['LOAD'] = n.allofact * (n.LOAD + n.bidMW)
        n['GENERATION'] = n.allofact * n.GENERATION
        n = n.drop(['allofact', 'bidMW'], axis=1)

        def swaperator(x):
            """If load is neg put in gen column and vice-versa - slow!"""
            if x.LOAD < 0.0:
                x["GENERATION"] = -x["LOAD"]
                x["LOAD"] = 0.0
            if x["GENERATION"] < 0.0:
                x["LOAD"] = -x["GENERATION"]
                x["GENERATION"] = 0.0
            return x
        n = n.apply(lambda x: swaperator(x), axis=1)

        # add/subtract dynamic loss from vSPD output
        # flow>0 flow "from bus"->"to bus": "from bus"=flow; "to bus"=flow-loss
        # flow<0 flow "to bus"->"from bus": "to bus"=flow; "from bus"=flow+loss
        # pdb.set_trace()
        bpos = b.copy().ix[b.FROM_MW >= 0.0]
        bneg = b.copy().ix[b.FROM_MW < 0.0]
        bpos['TO_MW'] = bpos['FROM_MW'] - bpos['DynamicLoss (MW)']
        bneg['TO_MW'] = bneg['FROM_MW']
        bneg['FROM_MW'] = bneg['FROM_MW'] + bneg['DynamicLoss (MW)']
        b = bneg.append(bpos)
        b = b.set_index(['tp', 'branch', 'FROM_ID_BUS', 'TO_ID_BUS'],
                        append=True)
        b = b.sort_index().ix[:, ['FROM_MW', 'TO_MW']]
        return n, b

    def trans_use(self, b, n, nmap=None, brmap=None, downstream=True):
        """Subroutine to calculate tranmission usage matrix"""
        def A(b, n, downstream=downstream):
            """Given branch flows and load/generation build the A matrix
            Note: similar to Amatrix.m """

            b = b.ix[:, ['FROM_MW', 'TO_MW']]  # grab the columns of interest
            allbus = list(set(b.index.levels[0]) | set(b.index.levels[1]))
            totbus = len(allbus)
            pg = n.GENERATION.ix[allbus].fillna(0.0).groupby(level=0).sum()
            pl = n.LOAD.ix[allbus].fillna(0.0).groupby(level=0).sum()
            A = pd.DataFrame(np.identity(totbus), index=allbus, columns=allbus)\
                .fillna(0.0)
            if downstream:
                """Build Nett downstream Ad with inflows"""
                b1 = b.ix[b.FROM_MW < 0].FROM_MW  # FROM_MW neg, flow into bus
                b2 = b.swaplevel(0, 1).sort()     # swap levels and sort
                b2 = -b2.ix[b2.TO_MW > 0].TO_MW   # TO_MW is pos, flow into bus
            else:
                """Build Gross upstream Au with outflows"""
                b1 = b.ix[b.TO_MW < 0].TO_MW  # FROM_MW neg, flow into bus
                b2 = b.swaplevel(0, 1).sort()     # swap levels and sort
                b2 = -b2.ix[b2.FROM_MW > 0].FROM_MW   # TO_MW pos, flow into bus
            b2.index.names = ['FROM_ID_BUS', 'TO_ID_BUS']  # to match flow dir
            cji = b1.append(b2).sortlevel()  # append and sort
            cji = cji.groupby(level=[0, 1]).sum()  # sum parrallel branches
            # calculate Nodal through-flows using inflows
            inflow1 = -b.ix[b.FROM_MW < 0].groupby(level=0).sum().FROM_MW  # out
            inflow2 = b.ix[b.TO_MW > 0].groupby(level=1).sum().TO_MW
            inflow = pd.DataFrame({'inflow1': inflow1,
                                   'inflow2': inflow2}).fillna(0.0).sum(axis=1)
            # calculate Nodal through-flows using outflows
            outflow1 = b.ix[b.FROM_MW > 0].groupby(level=0).sum().FROM_MW
            outflow2 = -b.ix[b.TO_MW < 0].groupby(level=1).sum().TO_MW
            outflow = pd.DataFrame({'outflow1': outflow1,
                                    'outflow2': outflow2}).fillna(0.0)\
                .sum(axis=1)
            Pi = pd.DataFrame({'inflow': inflow, 'outflow': outflow})\
                .fillna(0.0)
            PI = pd.DataFrame({'Pi+pg': Pi.inflow.add(pg, fill_value=1)
                               .fillna(0.0),
                               'Pi+pl': Pi.outflow.add(pl, fill_value=0)
                               .fillna(0.0)})
            if downstream:
                """Build Nett downstream Ad with inflows"""
                for i, r in pd.DataFrame({'cji': cji}).iterrows():
                    v = r.values/(PI['Pi+pl'].ix[i[0]])
                    A.ix[i[1], i[0]] = v
            else:
                """Build Gross upstream Au with outflows"""
                for i, r in pd.DataFrame({'cji': cji}).iterrows():
                    v = r.values/(PI['Pi+pg'].ix[i[1]])
                    A.ix[i[0], i[1]] = v

            A = A.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            iA = np.linalg.inv(A)  # invert
            Pi = PI['Pi+pg'].ix[allbus]
            if downstream:  # calculate the nett generation
                pgn = iA.dot(np.array(pl.values)) * np.array(pg.values) * \
                    np.array(((1/Pi).fillna(0.0)).values)
                pg = pd.Series(pgn, index=allbus).fillna(0.0)
            else:  # calculate the gross demand
                plg = iA.dot(np.array(pg.values)) * np.array(pl.values) * \
                    np.array(((1/Pi).fillna(0.0)).values)
                pl = pd.Series(plg, index=allbus).fillna(0.0)
            return A, iA, pg, pl, b2, cji, Pi

        def topo(iA, pi, bd, plg, nmap=nmap, downstream=True):
            """Calculate the topological distribution matrices"""
            if self.run == 'tpm':  # use self.nmap
                nmap = self.nmap
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
            b_inx = np.array([b_in, ] * totbus).transpose()
            b_outx = np.array([b_out, ] * totbus).transpose()
            iA_df = pd.DataFrame(iA, index=allbus, columns=allbus)
            i_A_ibus = iA_df.ix[ibus, :]
            i_A_jbus = iA_df.ix[jbus, :]
            Pdd = [plg.values, ] * len(b_in)
            Dilk1 = np.abs(b_outx) * i_A_jbus.values * Pdd * (1 / pij)
            Dilk2 = np.abs(b_inx) * i_A_ibus.values * Pdd * (1 / pii)
            if downstream:  # calculate nett branch flows downstream to demand
                df = Dilk1 * bnegx + Dilk2 * bposx
            else:  # calculate gross branch flows for upstream to generators
                df = Dilk1 * bposx + Dilk2 * bnegx
            df = pd.DataFrame(df, index=bdd.index, columns=allbus).fillna(0.0)
            idx = list(set(df.columns) & set(nmap.index))  # filter
            df = df[idx]
            # print df.head()
            return df

        def bustonode(df, brmap=None, nmap=None):
            """this function, given the trace solution at bus number level, maps
            the bus number data to node data.  brmap maps branch FROM/TO bus
            numbers to circuit names, while nmap maps buses to the common node
            GXP/GIP names.   Both brmap and nmap are required for running the
            trace in vspd mode.  In tpm node these are not required as this
            run used a single fixed grid."""
            if self.run == 'tpm':
                brmap = self.brmap
                nmap = self.nmap
            # print brmap.head()
            # print nmap.head()

            # reindex no node labels
            # def mapper(x, mapseries):
            #     """attempt to map node names to bus numbers"""
            #     try:
            #         print mapseries[x]
            #         return mapseries[x]
            #     except:
            #         return x
            # df.index = df.index.map(lambda x: mapper(x, brmap))
            # df.columns = df.columns.map(lambda x: mapper(x, nmap))
            # print brmap.head()
            # print nmap.head()
            # print df.head()
            # pdb.set_trace()
            # df.columns = df.columns.map(lambda x: nmap[x])
            df.index = df.index.map(lambda x: brmap[x])
            df = df.rename(columns=nmap.to_dict())
            # print df.head()
            # pdb.set_trace()
            # select rows and columns that sum to greater than 0
            df = df.ix[df.sum(axis=1) > 0, df.sum() > 0]
            # sum columns to the node label level
            df = df.groupby(level=0, axis=1, sort=False).sum()

            def seriestolist(x):
                """We get pd.Series opjects when mapping to branch name level,
                objects from series to array types"""
                if isinstance(x, type(pd.Series())):
                    idx = x.values
                    return ', '.join(str(e) for e in idx)
                else:
                    return x

            df.index = df.index.map(lambda x: seriestolist(x))
            # print df.head()
            # pdb.set_trace()
            return df

        if downstream:  # net downstream pg is nett of losses, pl is actual
            Ad, iAd, pg, pl, bdd, cjid, Pi = A(b, n, downstream=downstream)
            df = topo(iAd, Pi, b, pl, nmap=nmap, downstream=True)  # bus level trace
            df1 = bustonode(df.copy(), brmap=brmap, nmap=nmap)  # node level trace
        else:  # gross upstream pg is actual, pl grossed with losses
            Au, iAu, pg, pl, bdu, cjiu, Pi = A(b, n, downstream=downstream)
            df = topo(iAu, Pi, b, pg, downstream=False)  # bus level trace
            df1 = bustonode(df.copy(), brmap=brmap, nmap=nmap)  # node level trace

        return df, df1, pl, pg

    def sub_usage(self, df, pl, pg, nmap=None):
        """Calculate substation usage matrix.  Groups to node level, then substation
        level.  MW values are summed at substation level so represent total
        through flow, summed through all buses that comprise a substation.  As a
        substation consists of multiple buses, the resulting flows are not very
        representative or useful for anything other than calculating usage.
        There are a number of other methods that could be used, i.e, select
        the highest voltage bus.
        OR,
        (9/7/2015) use the td/tu arrays taking mean flows into and out of each
        substation.  This could be a better solution and potentially be
        implemented with a regular expression matching in a post process step to
        determine the circuits attached to each substation (DJH).
        """
        if self.run == 'tpm':
            nmap = self.nmap

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

        def bus2node(df, nmap):
            """Given bus level data, group up to node level"""
            # df.index = df.index.map(lambda x: nmap[x])
            # df.columns = df.columns.map(lambda x: nmap[x])
            df = df.rename(columns=nmap.to_dict(), index=nmap.to_dict())

            return df

        b_usage = bus_trace_usage(df, pl, pg, nmap)
        n_usage = bus2node(b_usage, nmap)

        return n_usage

    def output_results(self, df, ymd, tt):
        """function that outputs trace results based on trace type (tt)"""
        if tt == 'fc':
            pd.Series(self.fc).to_csv(os.path.join(self.outpath, 'fc',
                                                   'fc' + ymd + '.csv'))
        else:
            if not self.tp:
                pac = os.path.join(self.outpath, 'd', tt + '_' + ymd + '.csv')
                P = pd.Panel(df).fillna(0.0)
                P.mean(0).to_csv(pac, float_format='%.4f')
                logger.info("|OUTPUT: " + pac + '|')
            if self.tp:
                pap = os.path.join(self.outpath, 'tp',
                                   tt + '_' + ymd[:12] + '.csv')
                df.to_csv(pap)
                logger.info("|OUTPUT: " + pap + '|')

    def trace_day(self, n, b):
        """do the daily trace thing"""
        for i, dt in enumerate(n.index.levels[0]):
            tu = {}
            td = {}
            su = {}
            sd = {}
            try:
                info_text = "TRACE: For " + str(dt)
                logger.info(info_text)
                n2 = n.xs(dt, level=0)
                nmap = n2.reset_index().ix[:, ['Bus', 'Node']].set_index('Bus').Node
                # print nmap
                # pdb.set_trace()
                n2 = n2.reset_index('Node', drop=True)
                b2 = b.xs(dt, level=0)
                brmap = b2.reset_index().ix[:, ['branch', 'FROM_ID_BUS',
                                                'TO_ID_BUS']].set_index(['FROM_ID_BUS', 'TO_ID_BUS']).branch
                b2 = b2.reset_index('branch', drop=True)
                # print n2.head()
                # print b2.head()
                # perform downstream trace
                dfd, dfd1, pl, pg = self.trans_use(b2, n2, nmap=nmap,
                                                   brmap=brmap,
                                                   downstream=True)
                dfds = self.sub_usage(dfd, pl, pg, nmap=nmap)
                td[str(dt)] = dfd1
                sd[str(dt)] = dfds
                # upstream trace
                dfu, dfu1, pl, pg = self.trans_use(b2, n2, nmap=nmap,
                                                   brmap=brmap,
                                                   downstream=False)
                dfus = self.sub_usage(dfu, pl, pg, nmap=nmap)
                tu[str(dt)] = dfu1
                su[str(dt)] = dfus
                self.fc[(dt)] = 0
            except Exception:
                logger.error("***FAILED*** for " + str(dt) + "***")
                logger.error(traceback.print_exc())
                self.fc[(dt)] = 1
                pass
            if self.tp:
                self.output_results(dfd1, self.ymd + str(i+1).zfill(2), 'td')
                self.output_results(dfu1, self.ymd + str(i+1).zfill(2), 'tu')
                self.output_results(dfds, self.ymd + str(i+1).zfill(2), 'sd')
                self.output_results(dfus, self.ymd + str(i+1).zfill(2), 'su')

        self.output_results(td, self.ymd, 'td')
        self.output_results(tu, self.ymd, 'tu')
        self.output_results(sd, self.ymd, 'sd')
        self.output_results(su, self.ymd, 'su')
        self.output_results(su, self.ymd, 'fc')


    def trace_month(self):
        """do the month trace thing"""
        for m in pd.date_range(start=self.start, end=self.end, freq='M'):
            td = {}  # downstream transmission usage
            sd = {}  # downstream substation usage
            tu = {}  # upstream transmission usage
            su = {}  # upstream substation usage
            ym = str(m.year) + str(m.month).zfill(2) + '.csv'
            vSPD_b = os.path.join(self.inpath, 'b_' + ym)
            vSPD_n = os.path.join(self.inpath, 'n_' + ym)
            info_text = 'INPUT: b_' + ym + ', n_' + ym
            logger.info(info_text)
            n, b = self.load_tpm_vSPD_data(vSPD_b, vSPD_n)
            # pdb.set_trace()
            for day in n.index.levels[0]:
                ymd = str(m.year) + str(m.month).zfill(2) + str(day.day).zfill(2) + '.csv'
                for tp in n.index.levels[1]:
                    info_text = "TRACE: " + str(day.date()) + "|TP " + \
                                str(int(tp)).zfill(2)
                    logger.info(info_text)
                    try:
                        # get TP level data
                        n2 = n.xs(day, level=0).xs(tp, level=0)
                        n2 = n2.reset_index('node', drop=True)
                        b2 = b.xs(day, level=0).xs(tp, level=0)
                        b2 = b2.reset_index('branch', drop=True)
                        # Perform downstream trace
                        dfd, dfd1, pl, pg = self.trans_use(b2, n2,
                                                           downstream=True)
                        dfds = self.sub_usage(dfd, pl, pg)
                        td[(str(tp))] = dfd1
                        sd[(str(tp))] = dfds
                        # Perform upstream trace
                        dfu, dfu1, pl, pg = self.trans_use(b2, n2,
                                                           downstream=False)
                        dfus = self.sub_usage(dfu, pl, pg)
                        tu[(str(tp))] = dfu1
                        su[(str(tp))] = dfus
                        self.fc[(day, str(tp))] = 0
                    except Exception:
                        logger.error("***FAILED*** for " + str(day) +
                                    " trading period " + str(int(tp)) + "***")
                        logger.error(traceback.print_exc())
                        self.fc[(day, str(tp))] = 1
                        pass

                self.output_results(td, ymd, 'td')
                self.output_results(tu, ymd, 'tu')
                self.output_results(sd, ymd, 'sd')
                self.output_results(su, ymd, 'su')

        pd.Series(self.fc).to_csv(os.path.join(self.outpath, 'fc.csv'))

###############################################################################
# Start TRACE
#
# For each TP;
#   - perform UP stream trace for generation usage of transmission assets
#   - perform DOWN stream trace for load/demand usage of transmission assets.
#
###############################################################################

if __name__ == '__main__':
        t = trace(p.t)  # run instance
        t.create_output_dir()

        if t.run == 'tpm':  # if TPM Concept Consulting run
            logger.info(33*'*')
            logger.info("Start TPM tracing routine".center(33))
            logger.info(33*'*')
            t.trace_month()

        if t.run == 'vspd':  # if we are tracing the output of a vSPD run
            # this is a daily run where self.start is considered to be the day
            logger.info(40*'*')
            logger.info("Start vSPD tracing routine".center(40))
            logger.info(40*'*')
            if t.start < datetime(2009, 7, 21):
                n, b = t.load_daily_vSPD_data(pre_msp=True)
                t.trace_day(n, b)
            if t.start > datetime(2009, 7, 21):
                n, b = t.load_daily_vSPD_data(pre_msp=False)
                t.trace_day(n, b)

        if t.run == 'testA':
            logger.info(40*'*')
            logger.info("Start 3-bus test tracing routine".center(40))
            logger.info(40*'*')
            # pdb.set_trace()
            t.trace_month()

        if t.run == 'testB':
            logger.info(40*'*')
            logger.info("Start 4-bus test tracing routine".center(40))
            logger.info(40*'*')
            t.trace_month()



