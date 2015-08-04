Files
temp
vSPDcase               / "vSPDcase.inc" /
TradePeriodNodeBus     / "TradePeriodNodeBus.csv" /
DateTimeNodeBus        / "DateTimeNodeBus.csv" /
FileNameList           / "..\vSPD\Programs\vSPDfileList.inc" /
;

Set
iFile 'set of gdx file name'

$include ..\vSPD\Programs\vSPDfileList.inc
;

DateTimeNodeBus.pc = 5 ;
DateTimeNodeBus.lw = 0 ;
DateTimeNodeBus.pw = 9999 ;
DateTimeNodeBus.ap = 0 ;
put DateTimeNodeBus;
put 'DateTime', 'Node', 'Bus', 'Allo' / ;
putclose;


loop(iFile,
*  Create the file that has the name of the input file for the current case being solved
   putclose vSPDcase "$setglobal  vSPDinputData  " iFile.tl:0;
*  Solve the model for the current input file
   put_utility temp 'exec' / 'gams extract1' ;
);

*execute 'rm "*.~gm"' ;
*execute 'rm "*.lxi"' ;
*execute 'rm "*.log"' ;
*execute 'rm "*.put"' ;
*execute 'rm "*.lst"' ;
execute 'rm "vSPDcase.inc"' ;


