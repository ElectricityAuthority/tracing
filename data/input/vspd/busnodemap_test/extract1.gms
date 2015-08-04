* Include some settings
$include vSPDcase.inc
* point to walters gdx grab directory that contains all gdx files ever...
$if not exist "z:\home\humed\walter_home\vSPD\gdx_grab\extracted\%vSPDinputData%.gdx" $goto nextInput
Files
  TradePeriodNodeBus             / "TradePeriodNodeBus.csv" /
  DateTimeNodeBus                / "DateTimeNodeBus.csv" /
;

Sets
i_DateTime                   ''
i_TradePeriod                ''
i_Node                       ''
i_Bus                        ''
i_DateTimeTradePeriodMap(i_DateTime,i_TradePeriod ) 'Node bus mapping by trading period';
* i_TradePeriodNodeBus(i_TradePeriod ,i_Node,i_Bus)     'Node bus mapping by trading period'
Parameters
i_TradePeriodNodeBusAllocationFactor(i_TradePeriod ,i_Node,i_Bus)     'Node bus allocation factor by trading period'
;

* $gdxin "..\vSPD\Input\%vSPDinputData%.gdx"
$gdxin "z:\home\humed\walter_home\vSPD\gdx_grab\extracted\%vSPDinputData%.gdx"
$load i_DateTime i_TradePeriod i_Node i_Bus
* $load i_DateTimeTradePeriodMap i_TradePeriodNodeBus i_TradePeriodNodeBusAllocationFactor
$load i_DateTimeTradePeriodMap i_TradePeriodNodeBusAllocationFactor
$gdxin

*TradePeriodNodeBus
DateTimeNodeBus.pc = 5 ;
DateTimeNodeBus.lw = 0 ;
DateTimeNodeBus.pw = 9999 ;
DateTimeNodeBus.ap = 1 ;
put DateTimeNodeBus;
loop[ (i_DateTime,i_TradePeriod,i_node,i_Bus) $ (i_TradePeriodNodeBusAllocationFactor(i_TradePeriod ,i_Node,i_Bus) and i_DateTimeTradePeriodMap(i_DateTime,i_TradePeriod )),
   put i_DateTime.tl, i_Node.tl, i_Bus.tl,  i_TradePeriodNodeBusAllocationFactor(i_TradePeriod, i_Node, i_Bus)/ ;
] ;

$label nextInput
