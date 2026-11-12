"""Norms of the linear IMF residual/component operators on the notebook design.
Periodic grid n=1000, windows 501..31 (a=sqrt2), circulant smoothers W_k.
Reports ||I-W_k||_2, ||prod_{j<k}(I-W_j)||_2 and _inf, ||W_k prod_{j<k}(I-W_j)||_inf."""
import numpy as np, csv
n=1000; wins=[501,355,251,177,125,89,63,45,31]
def impulse(win,kern):
    R=(win-1)//2; d=np.arange(-R,R+1); u=d/R
    w=np.clip((1-np.abs(u))**2 if kern=='sq' else 1-u**2,0,None); w/=w.sum()
    h=np.zeros(n); h[d%n]+=w; return h
rows=[]
for kern in ['sq','epa']:
    W=[np.fft.fft(impulse(w,kern)).real for w in wins]
    for k in range(len(W)):
        P=np.prod([1-W[j] for j in range(k)],axis=0) if k else np.ones(n)
        rows.append(dict(kernel={'sq':'(1-|u|)_+^2','epa':'Epanechnikov'}[kern],k=k+1,window=wins[k],
            norm2_I_minus_Wk=round(np.max(np.abs(1-W[k])),4),
            norm2_prod=round(np.max(np.abs(P)),4),
            norminf_prod=round(np.abs(np.fft.ifft(P).real).sum(),3),
            norminf_component=round(np.abs(np.fft.ifft(W[k]*P).real).sum(),3)))
with open('linear_contraction_check.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
for r in rows: print(r)
