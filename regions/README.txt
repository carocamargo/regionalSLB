Regional Budget Steps

Objective 1: Regional Clusters Identification
	---> deltaMAPS
	---> SOM

For this we first need to get working data and pre-process it.

	- SLB-clustering-data-preprocess.py
		In this script we pre-process the altimetry CMEMS dataset.
		Pre-processing is contained on:
			1. remove trend
			2. remove seasonality
			3. apply spatial gaussian filter of 300km
			4. regrid to 1 degree (original resolution is 0.25 deg)
		Most files are saved on '/Volumes/LaCie_NIOZ/reg/world/data/'

	- SLB-get_components.py
		In this script we get the SLA from altimetry (not processed, so with full signal), steric and barystatic from the different folders, and set it all to the same working folder. 
		We ensure all datasets are in the same spatial and temporal resolution and have the same units.
		All SLAs and trends are given in METERS and METERS/year at this moment. 

	- SLB-landmask.py
		Create a landmask based on the steric and altimetry data. 
		That is: covering from -66 to 66 degrees latitude. 

	- SLB-merge_components.py
		Combine altimetry, steric (ENS) and barystatic (IMB+WGP) time series and trends (from 1993-2017, for alt and steric and 1993-2016 for bary) into one single file.
		Time series run from jan 1993-dec 2017
		All units are in MILIMETERS and MILIMETERS/year 

	- SLB-clustering-data_preprocess-compoenents.py
		To run SOM with steric and mass, we get the steric and barystatic SL stored in the SLB-merge_componenets.py script, and apply pre-processing:
			1. remove trend
			2. remove seasonality
			3. apply spatial gaussian filter of 300km
		It is not necessary to regrid, because the data is already in 1deg
		Files are saved on '/Volumes/LaCie_NIOZ/reg/world/data/'
		
		--> the final files from 1993-2016 used for SOM clustering were then copied to the subfolder 'comps' ('/Volumes/LaCie_NIOZ/reg/world/data/comps/'), and transfered to Harrier 

	Up to here, most files are saved at '/Volumes/LaCie_NIOZ/reg/world/data/'
	Some local copies (of the preprocessed data to run SOM and of the merged budget components) on the folder /Users/ccamargo/Desktop/budget/data/


Now we have all data, we can start clustering.
- DeltaMAPS
	- To find the ideal k, we ran deltaMAPS with size o neighrbourhood (k) varying from 4 to 24, with the input data regrided to 1 and 2 degrees.
	- We used the detrended, deseasonalized and smoothed (300km gaussian filter) sea-level variabilities to perform the clustering.
	- The computatinal time increases significantly from 2 degrees resolution to 1 degree. 
	- Computations ran on the NIOZ cluster with the scripts:
		- dmaps-loop-world.ipynb (for the 1 degree resolution)
		- dmaps-loop-world-res2.ipynb (for 2 degrees resolution)
	- The outputs were saved on the folders /world and /world2 (for 1 and 2 degrees, respectively).
	- These folders were then transfered to the HD drive, on the directory:
		/Volumes/LaCie_NIOZ/reg/world/dmaps/
	- The outputs were then plotted with the script plot_dMAPS.py and saved on the plots/ folder. We ploted the clustering maps, the homogeneity map and the NMI matrix. 
	- The NMI matrix can be used to find the ideal k. Ideally, you want a K which is surrounded by other high correlation valyes, so that if you vary the size of the neighborhood, your clustering won't suffer large changes. 
	- Looking at the clusters and the NMI matrix, we decided that k=5 and k=7 yielded the best clusters for our analysis. Other suitable Ks according to the NMI matrix would be: K~14 and K~19. When K varies from 19 to 21, we have good clustering, but more regions that were not grouped. k=5 and k=7 seemed to be the best results. 
	-Folders k5 and k7 were copied from '/Volumes/LaCie_NIOZ/reg/world/dmaps/world/' to working folder '/Users/ccamargo/Desktop/budget/regions/deltamaps/'

	Re-did the degree 1 again, ebcause before our dimensions was 361x181. So we regridded again, run in the cluster, out put saved on '/Volumes/LaCie_NIOZ/reg/world/dmaps/world_v2/'
	- plot it with script plot_dMAPS_v2.py
	
- SOM
	- MiniSOM - Python
		A lot of tests were performed to find the ideal configurations of MiniSom. 
		Those working scripts are found in the py_scripts folder (MOVE TO A SUBFOLDER NAMED SOM!!)
		Finally, we use the following configuration: 
			- Data is normalized by range
			- PCA initialization
			- Hexagonal topology
			- Gaussian neighborhood function
			- Random training with seeds=10
			- Training length of 1000000 (for quicker tests n = 10000)
			- sigma = 1 (default)
			- learning rate = 0.5 (default)
			- map size: 3x3, 4x4, 5x5

	- SOM ToolBox - MatLab
		- Ran on IMEDEA cluster harrier. 
		- saved output cluster map on netcdfs:
			- cluster directory: /home/cmachado/data/som_output/
			- local directory: /Volumes/LaCie_NIOZ/reg/world/som/matlab/

- Cluters Mask
	Unite the clusters form deltmaps and from SOM into a single file

Now we have some clusters, we can do some 'quick' budget analysis:
- Preliminary Budget: budget_exploratory_plots.py
Budget on SOM regions:
	-plot-SOM_maps-matla_outputs_2-Copy2.ipynb
	In this script we make the budget on different clusters for different SOM maps
	- corr_heatmap-Clusters.ipynb : plot a correlation heat map between the clusters




