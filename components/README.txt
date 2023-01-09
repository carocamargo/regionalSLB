Sea-level Budget Components

Let's organize the data for the budget:

1. Altimetry: 
SLB-altimetry.py 
	- Get different altimetry datasets: AVISO (SSALTO/DUACS), SLcii; MeASUReS (JPL); CMEMS; CSIRO
	- Regrid all files to 180x360 

 SLB-altimetry-part2.py
 	- Merge all datasets into a single dataset
 	- Select since 1993
 	- Compute Ensemble mean

 SLB_alt_trend.py  
 	- run Hector on altimetry trend (cluster, see script ALT-hector.py) 
	*** UPDATE: FOR ALL PERIODS AND DATASETS (waiting for hector runs still)
 	- periods: 
 		a. 1993-2017
 		b. 1993-2016
 		c. 1993-2019 (or 2020?)
 		d. 2003-2016
 		e. 2005-2015 
 	- select best noise model combination


2. Barystatic: 
Barystatic SL is already in trends, because we run Hector on the mass source changes. (See scripts on OM/revisions folder)
Trend periods:
	a. 2005-2015
	b. 2003-2016
	c. 1993-2017

	2b-run_SLE-ASL.py (in the OM/revisions/ folder)
	- Run SLE and get the ASL, instead of RSL (which is what we used for the ESD manuscript)
	- Combine with uncertainties from ESD manuscript (make final dataset)

	SLB-barysatic.py
	- Select total barystatic trend and uncertainty for the 6 combos (JPL, CSR, IMB+WGP, IMB+GWB+ZMP, UCI+WGP, UCI+WGP+ZMP)
	- Compute ensemble mean 
	
	SLB-barystatic_SLF_timestep.py
	- Run SLE for each time step for all datasets (from mass change in source)
	- Combine it all in one dataset

3. Steric:
SLB-steric.py
	- Combine all time series in one file

SLB-steric_trends.py 
	- Get already compiled and published trends (JGR:Oceans manuscript)
	- Save selected noise models with sel_and_save_prefer_NM_steric-all_datases.py
	- Trend periods:
		a. 2005-2015
		b. 1993-2017

SLB-steric-deep.py
	- Combine deep steric from Purkey & Johnson (2010-updated) (and Chang et al (2019) in a single file)
	- These are already trends. P&J: 1990-2016(?) - assume linear trend

SLB-steric_full.py
	- Add deep steric to 0-2000m steric


4. Dynamic (REA)
dyn_sl.py (PREP part 1)
	- Get original ocean reanalysis files (drive SLB)
	- Regrid to 180x360
	- Combine it into a single dataset (save it at LaCie_NIOZ drive)
dyn_sl-part2.py
	- Remove ARMOR and SODAv4.3.2. from the file
	- Compute Ensemble mean
	- Apply mask for polar regions
	- Correct suddent drop C-GLORS
	- Remove 2020 (only ARMOR was this long, we dont have data for the other ones )

SLB-dynamic.py (PREP part 2)
	- Get ocean reanalysis SSH (regrided) time series
	
	-  get regional steric (full depth)
	- compute steric anomaly
	- compute dynamic SL: Dynamic Sl = SSH' - SSL'
	 where SSH' = SSH(t,x,y) - SSH(t) of the reanalysis, 
	 and SSL' = SSL(t,x,y) - SSL(t) of full depth steric sea level
	 - compute sterodynamic SL: Sterodynamic SL = SSH' + SSL(t)
	 where SSL(t) is the time varying global mean steric sea level. 

SLB-dynamic_trend.py 
	*** UPDATE: FOR ALL PERIODS AND DATASETS (waiting for hector runs still)

	- Compute trends with hector on cluster (see script DSL-hector.py)
	- Select best NM and save it 

5. Manometric (GRACE)
SLB_manometric.py
	- Get ocean mass signal from grace JPL and CSR Mascons (manometric SL)
	- get barystatic SL (fingerprints) in Absolute Sl ( not RSL)
	- Compute trend of manometric
	- remove barystatic from manometric, obtaining dynamic component observed by GRACE. 

SLB-manometric_trend.py *****TO DO******
	- Compute trends with hector on cluster (see script mSL-hector.py)
	- Select best NM and save it 


Scripts to re-run if we change something
on altimetry:
	- SLB-altimetry-part2.py
on steric (upper):
	-SLB-steric.py
	-SLB_steric_full
	-SLB_dynamic
on steric (deep):
	-SLB-deep.py
	-SLB_steric_full
	-SLB_dynamic
on steric (full):
	-SLB_steric_full
	-SLB_dynamic



