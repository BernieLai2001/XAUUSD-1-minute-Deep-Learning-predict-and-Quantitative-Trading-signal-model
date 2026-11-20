# 按月份（美东）拆解止损比例上升：点差相对止盈止损距离（s/d）、当月涨跌
import pandas as pd
from gold_dl.config import path
from gold_dl.labels import FOCUS_H, K_DEFAULT, RR_DEFAULT, TICK, sl_dist
H = FOCUS_H   # config.yaml 的 report.focus_holding_min
YL, YS = f'long_H{H}', f'short_H{H}'
l=pd.read_parquet(path('labels','labels_default.parquet'))
f=pd.read_parquet(path('features','features_1min.parquet'),columns=['raw_atr','raw_close','raw_high','raw_low'])
d=l.join(f); d=d[d.pre_roll==0]
rng=(d.raw_high-d.raw_low)
fut=rng[::-1].rolling(H,min_periods=H*2//3).mean()[::-1].shift(-1)
d['ratio']=fut/d.raw_atr
D=sl_dist(d.raw_atr,H,K_DEFAULT); s=d.spread_ticks*TICK
d['s_over_d']=s/D
d['m']=(d.index - pd.Timedelta(hours=4)).to_period('M')
g=d.groupby('m')
out=pd.DataFrame({
 'split':g.split.agg(lambda x:x.mode()[0]),
 'n':g.size(),
 'L_SL':g[YL].apply(lambda x:(x==1).mean()*100),
 'L_TP':g[YL].apply(lambda x:(x==0).mean()*100),
 'S_SL':g[YS].apply(lambda x:(x==1).mean()*100),
 'TO':g[YL].apply(lambda x:(x==2).mean()*100),
 'price':g.raw_close.median(),
 'atr':g.raw_atr.median(),
 'atr_bp':(g.raw_atr.median()/g.raw_close.median()*1e4),
 'spr_tick':g.spread_ticks.median(),
 's/d%':g.s_over_d.median()*100,
 'fut/atr':g.ratio.median(),
})
# 随机游走理论：碰线条件下 P(SL)=(rr·d+s)/((1+rr)·d)，盈亏比 1:1 时为 (d+s)/(2d)
d['pSL']=(RR_DEFAULT+d.s_over_d)/(1+RR_DEFAULT)
out['理论SL/(SL+TP)']=g.pSL.mean()*100
hit=d[YL].isin([0,1])
out['实际SL/(SL+TP)']=d[hit].groupby('m')[YL].mean()*100
print(out.round(2).to_string())
c=g.raw_close
out2=pd.DataFrame({'月涨跌%':(c.last()/c.first()-1)*100,'多SL-空SL':out.L_SL-out.S_SL,'平均SL':(out.L_SL+out.S_SL)/2,'TO':out.TO,'s/d%':out['s/d%']})
print(out2.round(1).to_string())
print('corr(月涨跌, 多SL-空SL)=',round(out2.iloc[1:,0].corr(out2.iloc[1:,1]),2))
print('corr(s/d, 平均SL)=',round(out2.iloc[1:,4].corr(out2.iloc[1:,2]),2))
