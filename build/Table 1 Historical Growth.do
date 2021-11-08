* do-file for Table 1: Decomposition of growth in health expenditures
clear all 
capture log close
set more off

* change your path to root of replication dir
capture cd ~/cedia/Projets/agghealth_replicate

* use OECD health data extract
use data_sources/oecd_history.dta, clear
sort country year

merge 1:1 country year using data_sources/leatbirth.dta

* Country Names -numbers
gen str2 cshort = "DK" if cid==19
replace cshort = "FR" if cid==16
replace cshort = "DE" if cid==12
replace cshort = "IT" if cid==17
replace cshort = "NL" if cid==14
replace cshort = "SP" if cid==15
replace cshort = "SE" if cid==13
replace cshort = "UK" if cid==9
replace cshort = "US" if cid==10
label var cshort "short country name"

* keep countries, IT is out because data starts in 1970
keep if inlist(cid,10,12,13,14,15,16,19)

* keep years
global start_yr = 1970
global stop_yr = 2007
keep if year>=$start_yr
keep if year<=$stop_yr

*interpolate france (missing data in some years, no material effect on decomposition)
ipolate tothlthcpoecdcap year if cshort=="FR", gen(tothlthcpoecdcap1)
replace tothlthcpoecdcap = tothlthcpoecdcap1 if cshort=="FR"

* set data as panel
sort cid year
tsset cid year


* The health care var and GDP are in NCU. This does not matter as we do fixed effects
* we take logs
gen pmreal_ma = tothlthcpoecdcap
gen lmr=log(pmreal_ma)
gen yreal_ma  = gdp15ncucap
gen ly=log(yreal_ma)
gen age65p_ma  = log(propop65p)

* we create splines for decades (last one has 8 years)
mkspline year_70 1979 year_80 1989 year_90 1999 year_00 = year
global yr_splines "year_*"

* just check how far spending was in 1970
gen eu = cid!=10
tabstat tothltcpoecdpppcap [aw=totpop] if inrange(year,1970,1972), by(eu)
tabstat tothltcpoecdpppcap [aw=totpop] if inrange(year,2005,2007), by(eu)

* set panel again
*tsset cid year

* XS
reg lmr ly age65p_ma if year==$stop_yr

* fixed effect, LSDV approach
reg lmr ly age65p_ma  $yr_splines i.cid

* within gives same thing
xtreg lmr ly age65p_ma  $yr_splines, fe

* test whether trend zero
testparm $yr_splines

* predict residual, including country fixed effect
predict resid, ue

* save income and age elasticity
scalar define epy=_b[ly]
scalar define epa=_b[age65p_ma]
scalar list epy epa

* do a simple computation of when unemployed growth was highest by g
egen decades = cut(year), at(1970(10)2010)
tab decades
gen dresid = d.resid
table cid decades, content(mean dresid)

* residual in levels to compute growth rates
replace resid = exp(resid)

* define periods, take 3 years to make base more stable
gen period = 0 if inrange(year,$start_yr,$start_yr+2)
replace period = 1 if inrange(year,$stop_yr-2,$stop_yr)

* perform computations of long-term growth rates and decompose

collapse pmreal_ma yreal_ma propop65p resid totpop leatbirth if period!=., by(cid period)

reshape wide pmreal_ma yreal_ma propop65p resid totpop leatbirth , i(cid) j(period)

global nyears = $stop_yr - $start_yr
foreach n in "pmreal_ma" "yreal_ma" "propop65p" "resid"  {
	gen `n'_change = (`n'1 / `n'0)^(1/$nyears) - 1
}

gen leatbirth_change = (leatbirth1/leatbirth0)-1

* these are the long-term growth rates in the variables
list cid *_change

* use coefficients from FE regs to compute implied change
gen eff_y = epy*yreal_ma_change
gen eff_p = epa*propop65p_change
gen eff_u = resid_change

* get the residual common trend
gen eff_r = pmreal_ma_change - eff_y - eff_p - eff_u

* combine for reporting the common trend and the residual
gen eff_ur = eff_r + eff_u

* compute cumulative effect of residual over period
gen cum_eff_ur = (1 + eff_ur)^$nyears-1

* report decomposition results 
list cid eff_*

* report cumulative effect
list cid cum_eff_*
spearman cum_eff_* leatbirth_change


* report cumulative effect
list cid cum_eff_* leatbirth_*

* compute EU averages using totpop at end as weight
gen eu = cid!=10
tabstat eff_* [aw=totpop1], by(eu)
tabstat cum_eff_* [aw=totpop1], by(eu)


exit



