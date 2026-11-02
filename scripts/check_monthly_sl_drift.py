# 按月份（美东）拆解止损比例上升：点差相对止盈止损距离（s/d）、当月涨跌
import numpy as np, pandas as pd
l=pd.read_parquet('labels/labels_default.parquet')
f=pd.read_parquet('features/features_1min.parquet',columns=['raw_atr14','raw_close','raw_high','raw_low'])
d=l.join(f); d=d[d.pre_roll_3d==0]
rng=(d.raw_high-d.raw_low)
fut=rng[::-1].rolling(30,min_periods=20).mean()[::-1].shift(-1)
d['ratio']=fut/d.raw_atr14
D=2*d.raw_atr14*np.sqrt(2); s=d.spread_ticks*0.1
d['s_over_d']=s/D
d['m']=(d.index - pd.Timedelta(hours=4)).to_period('M')
g=d.groupby('m')
out=pd.DataFrame({
 'split':g.split.agg(lambda x:x.mode()[0]),
 'n':g.size(),
 'L_SL':g.long_H30.apply(lambda x:(x==1).mean()*100),
 'L_TP':g.long_H30.apply(lambda x:(x==0).mean()*100),
 'S_SL':g.short_H30.apply(lambda x:(x==1).mean()*100),
 'TO':g.long_H30.apply(lambda x:(x==2).mean()*100),
 'price':g.raw_close.median(),
 'atr':g.raw_atr14.median(),
 'atr_bp':(g.raw_atr14.median()/g.raw_close.median()*1e4),
 'spr_tick':g.spread_ticks.median(),
 's/d%':g.s_over_d.median()*100,
 'fut/atr':g.ratio.median(),
})
# 随机游走理论：碰线条件下 P(SL)=(d+s)/(2d)
d['pSL']=(1+d.s_over_d)/2
out['理论SL/(SL+TP)']=g.pSL.mean()*100
hit=d.long_H30.isin([0,1])
out['实际SL/(SL+TP)']=d[hit].groupby('m').long_H30.mean()*100
print(out.round(2).to_string())
c=g.raw_close
out2=pd.DataFrame({'月涨跌%':(c.last()/c.first()-1)*100,'多SL-空SL':out.L_SL-out.S_SL,'平均SL':(out.L_SL+out.S_SL)/2,'TO':out.TO,'s/d%':out['s/d%']})
print(out2.round(1).to_string())
print('corr(月涨跌, 多SL-空SL)=',round(out2.iloc[1:,0].corr(out2.iloc[1:,1]),2))
print('corr(s/d, 平均SL)=',round(out2.iloc[1:,4].corr(out2.iloc[1:,2]),2))
