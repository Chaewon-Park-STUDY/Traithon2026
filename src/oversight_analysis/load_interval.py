# 검토 부하의 통계적 불확실성(Clopper–Pearson 95%)과 용량 역산. 입력은 레포 results/ 집계값만 사용.
import math
from scipy.stats import beta
def cp(k,n,a=0.05):
    lo = 0.0 if k==0 else beta.ppf(a/2,k,n-k+1); hi = 1.0 if k==n else beta.ppf(1-a/2,k+1,n-k); return lo,hi
S,N=0.278,1000
# 테스트 셋 비민감 표본 수: 전체 클릭베이트 1895, 정상 1337에서 정치(257,191)·경제(268,191) 제외
n_cb = 1895-257-268; n_ok = 1337-191-191
k_cb = round(0.030656934306569343*n_cb); k_ok = round(0.9570680628272251*n_ok)   # a+ 미포착(자동통과) 건수, a- 정상 자동통과 건수
print("n_cb,k_cb,n_ok,k_ok:",n_cb,k_cb,n_ok,k_ok, " 재현율검산:",k_cb/n_cb,k_ok/n_ok)
ap_lo,ap_hi=cp(k_cb,n_cb); an_lo,an_hi=cp(k_ok,n_ok)
print("a+ (클릭베이트 미포착) 95%%CP: %.4f [%.4f, %.4f]"%(k_cb/n_cb,ap_lo,ap_hi))
print("a- (정상 자동통과)   95%%CP: %.4f [%.4f, %.4f]"%(k_ok/n_ok,an_lo,an_hi))
def load(pi,ap,an): return S+(1-S)*(pi*(1-ap)+(1-pi)*(1-an))
print("\npi | 부하(점) | 부하 상한(통계구간) | 완벽모델 하한 | 미포착 건/일 점 [구간]")
for pi in [0.05,0.10,0.25]:
    pt=load(pi,k_cb/n_cb,k_ok/n_ok); up=load(pi,ap_lo,an_lo); fl=S+(1-S)*pi
    miss=(1-S)*N*pi
    print(f"{pi:.2f} | {pt*N:6.1f} | {up*N:6.1f} | {fl*N:6.1f} | {miss*k_cb/n_cb:.2f} [{miss*ap_lo:.2f}, {miss*ap_hi:.2f}]")
print("\n필요한 1인 일 처리량 c (건/인/일) = 부하 상한/R  [N=1000]")
for pi in [0.05,0.10,0.25]:
    up=load(pi,ap_lo,an_lo)*N
    print(pi, {R: round(up/R,0) for R in (2,3,4,5)})
print("\nR명이 감당하는 최대 일 기사 수 N_max = R*c/부하상한  (c=150 가정을 쓰지 않고 c를 열로)")
for pi in [0.05,0.10,0.25]:
    up=load(pi,ap_lo,an_lo)
    print(pi, {f"c={c}": {R: round(R*c/up) for R in (2,3,4)} for c in (100,150,200)})
# 일변동: N 고정 시 이관 건수 이항 변동
import math
pt=load(0.10,k_cb/n_cb,k_ok/n_ok)
print("\n일 이관 건수 이항 sd(π=.10,N=1000): %.1f"%math.sqrt(N*pt*(1-pt)))
