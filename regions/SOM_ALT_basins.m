clear all;close all;clc;
%% Script to compute temporal patterns and regionalization
%===============================================================
%        DESCRIPTION

% Self-Organizing Maps (SOM) analysis in the time domain
% Using Altimetry data from 1993-2016 ( monthly, 288 timesteps)
dir_folder=['/home/cmachado/data/comps/'];
ind=[2]; %altimetry treated data
file='ALT_FULL_TREAT';
% Data has been pre-processed: detrended, deseasonalized, smoothed 300km
% use the following SOM parameters:
% map size: [3,3] and [4,4]
% radius: [2,1] and [1,1]
% neigh functions: ep and gaussian
% training length 2,10
% normalization: by range and by var
%===============================================================

%===============================================================
%% set working path to local tool boxes:
path(path,'/home/cmachado/scripts/m_map') % mapping toolbox
path(path,'/home/cmachado/scripts/SOM-Toolbox/som') % SOM


%% open mask


basins={'ocean',...% 'indopacific_TM','atlantic_TM',
    'circ_ant',...
    'atlantic_noCC','indopacific_noCC',...
    'atlantic_full','indopacific_full'
    };
ii=1;
for bas=basins
    bas=char(bas)
    Filename =  '/home/cmachado/data/ocean_basins_mask_v6.nc';
    % ncdisp(Filename)
    ncid = netcdf.open(Filename,'NC_NOWRITE');
    mask = squeeze(ncread(Filename,bas));
    mask(mask>0)=1;
    %mask(mask==0)=NaN;
    not_basin = find(mask==0);
    netcdf.close(ncid)
    % mask=ones(360,180);

    %===============================================================
    % %          LOAD NETCDF DATA
    %===============================================================
    filesStruct = dir(strcat(dir_folder,'*.nc'));

    files = {filesStruct.name};
    files = files(ind);
    dimfiles=length(files);

    % get dimensions
    Filename = strcat(dir_folder,char(files(1)));
    ncid = netcdf.open(Filename,'NC_NOWRITE');
    lon=ncread(Filename,'lon'); lon=double(lon);
    lat=ncread(Filename,'lat'); lat=double(lat);
    time=ncread(Filename,'time'); time=double(time);
    dimlon=length(lon);  nx=dimlon;
    dimlat=length(lat);  ny=dimlat;
    dimtime=length(time); nt=dimtime;
    netcdf.close(ncid)
    % % loop over SLA
    SLA = zeros(nx*ny,nt,dimfiles);
    for i=1:dimfiles
        Filename = strcat(dir_folder,char(files(i)));
        ncid = netcdf.open(Filename,'NC_NOWRITE');
        % check dimensions
        % size(ncread(Filename,'sla'))

        data=squeeze(ncread(Filename,'sla'));
        netcdf.close(ncid)
        [nx,ny,nt] = size(data);
        % reshape to 2D: 
        SLA(:,:,i)=reshape(data,nx*ny,nt);%.* reshape(mask,nx*ny,1);
    end 
    % %
    [X Y]=meshgrid(lon,lat);

    lonlim=([min(min(X)) max(max(X))]);  lonlim=double(lonlim);
    latlim=([min(min(Y)) max(max(Y))]);  latlim=double(latlim);

    % =============================================================
    %     Removing pixels with NaNs correspondig to land points or not QC
    
    % add NaN to out of interest basin:
      SLA(not_basin,:,:)=NaN;
    if length(size(SLA))==2
        meanSLA = mean(SLA,2); % across time
    else % if we have more than one variable
        meanSLA = mean(SLA,3); % mean across variables;
        meanSLA = mean(meanSLA,2); % mean across time
    end
    
    
    inxnanmeanSLA=find(isnan(meanSLA)==1);

    %  if length(find((inxnanmeanU-inxnanmeanV)==0))==length(inxnanmeanV)
    m = isnan(meanSLA(:,1)); m1 = find(m==1); m2 = find(m==0);  %% position of NaNs and not-NaNs
    %  end
    % remove land
    SLA(m1,:,:)=[];

    % =============================================================
%     % set map projection for plotting
%     m_proj('robinson','longitudes',[1 360],'latitudes',[-70 70]);
%     % m_gshhs_c('save','gumby');
%     lonx=lon-180;
%     [XL, YL]=meshgrid(lonx, lat);

    % =============================================================
    % %            SOM ANALYSIS IN THE TIME DOMAIN
    % =============================================================
    % % Loop over changing parameters:
    ngbs = {'ep';'gaussian'};
    norms={'range','var'};
    location=0;

    x=3;
    y=x;
    msize = [x y]; % size of neural map
    for i_ngb=1:2
    ngb_function = char(ngbs(i_ngb));
    % ngb_function=char(ngbs(2));
    for n=[2,10]
    % n=2;
    
    sig=2;
    sigma=[2,1];
    norm='range';

    ncname = strcat('/home/cmachado/data/som_output/basins/', 'som_',num2str(x),'x',num2str(y), '_',file, '_sig_init',num2str(sig),'_norm_',norm, '_train_n',num2str(n),'_ngb_function_',ngb_function,'_mask_',bas,'_.nc');

    % % =============================================================
    %   PREP DATA
    clear sD sD1 sD2 sD3;
    % % =============================================================
    %   check if we are using 1 or 2 variables
    if dimfiles==1
        %  1) CONVERTION the data matrix into SOM matlab structure
        sD = som_data_struct(SLA); % only 1 variable
        % 2) Normalization
        sD = som_normalize(sD,norm);
    elseif dimfiles==2
        % 1) matlab Struct
        sD1 = som_data_struct(SLA(:,:,1));
        sD2 = som_data_struct(SLA(:,:,2));
        % 2) Normalization: 1 var per time
        sD1 = som_normalize(sD1,norm);
        sD2 = som_normalize(sD2,norm);
        % Concatenate variables for analysis
        D = [sD1.data,sD2.data];
        sD = som_data_struct(D);
        % normalize
        sD = som_normalize(sD,norm);
    else
        % 1) matlab Struct
        sD1 = som_data_struct(SLA(:,:,1));
        sD2 = som_data_struct(SLA(:,:,2));
        sD3 = som_data_struct(SLA(:,:,3));
        % 2) Normalization: 1 var per time
        sD1 = som_normalize(sD1,norm);
        sD2 = som_normalize(sD2,norm);
        sD3 = som_normalize(sD3,norm);
        % Concatenate variables for analysis
        D = [sD1.data,sD2.data,sD3.data];
        sD = som_data_struct(D);
        % normalize
        sD = som_normalize(sD,norm);
    end % if dimfiles

    % 3) INITIALIZATION
    clear sM;
    sM = som_lininit(sD,'msize',msize);

    % 4) TRAINING.
    [sM,sTr2]=som_batchtrain(sM,sD,'msize',msize,'tracking',0,'trainlen',n,'radius',sigma,'lattice','hexa','shape','sheet','neigh',ngb_function);

    % 5) Calculate the vector with the BMUs [bmus,qerrs] and errors
    % % where "bmus" == is the neuron more similar to the sample  and 
    % "qerrs" == quantizacion error associated to this BMU
    [bmus,qerrs] = som_bmus(sM,sD,1);
    nunits=sM.topol.msize(1,1)*sM.topol.msize(1,2);
    histo_ocurrencia_n=hist(bmus,nunits);
    prob_ocurrencia_n=(histo_ocurrencia_n/length(bmus))*100; %probability of occurrence of each pattern (neuron)

    bmu_map = zeros(nx*ny,1);
    bmu_map(m2) = bmus;
    bmu_map(m1) = NaN;
    bmu_map=reshape(bmu_map,nx,ny);

    [qe_n,te_n] = som_quality(sM,sD);
    % %
    % 6) DENORMALIZATION OF THE FINAL OUTPUT
    sM = som_denormalize(sM);
    if dimfiles==1
        sD = som_denormalize(sD);
    elseif dimfiles==2
        % denormalize each dataset separated, and then put it togehter again
        sD1 = som_denormalize(sD1);
        sD2 = som_denormalize(sD2);
        SLA = [sD1.data,sD2.data];
    sD = som_data_struct(SLA);
    else
         % denormalize each dataset separated, and then put it togehter again
        sD1 = som_denormalize(sD1);
        sD2 = som_denormalize(sD2);
        sD3 = som_denormalize(sD3);
        SLA = [sD1.data,sD3.data];
        sD = som_data_struct(SLA);
    end % if dimfiles

    % 7) Calculate the vector with the BMUs [bmus,qerrs] and errors
    [qe,te] = som_quality(sM,sD);
    [bmus,qerrs] = som_bmus(sM,sD,1);
    histo_ocurrencia=hist(bmus,nunits);
    prob_ocurrencia=(histo_ocurrencia/length(bmus))*100; %probability of occurrence of each pattern (neuron)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %  8)   Regions of same temporal variability
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear evolution_best_match
    clear h_temporal

    inx=isnan(sD.data);
    sD.data(inx)=0;

    npixels = length(m2);
    for k=1:npixels(1)
        h_temporal = som_hits(sM,sD.data(k,:));
        evolution_best_match(k)=find(h_temporal==1);
    end

    clear regions;
    regions = zeros(nx*ny,1);
    regions(m2) = evolution_best_match;
    regions(m1) = NaN;
    regions=reshape(regions,nx,ny);
% 
%     figure(ii);
%     clf
%     m_pcolor(X,YL,regions')
%     shading flat
%     colormap(jet(x*y));
%     %  m_usercoast('gumby','patch',[1 0.7 0.4]);
%     m_usercoast('gumby','patch',[0.4 0.4 0.4]);
%     m_grid('box','fancy','tickdir','out','fontsize',18);
%     h=colorbar('v','FontSize',18);
%     title(bas,'FontSize',18)
% %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %  9)   Extract the SOM patterns = modos
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear modos;
    modos=sM.codebook;
    modos=reshape(modos,nunits,nt,dimfiles);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %  10)  Save Output in NETCDF file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nvar=dimfiles;
    data_matrix =  sD.data';
    data_matrix(:,m1)=NaN;

    % reshape to [nvar,ntime,nx,ny]
    data_matrix = reshape(data_matrix,nt,nx,ny,nvar);

    %
    % % make netcdf 
    %initialize netcdf file
    % cd('/home/cmachado/data/som_output/');
    ncid = netcdf.create(ncname,'netcdf4');

    %define dimensions
    latdimID = netcdf.defDim(ncid,'lat',length(lat));
    londimID = netcdf.defDim(ncid,'lon',length(lon));
    timedimID = netcdf.defDim(ncid,'time',length(time));
    %depthdimID = netcdf.defDim(ncid,'depth',length(depth));
    neurondimID = netcdf.defDim(ncid,'neurons',nunits);
    idxdimID = netcdf.defDim(ncid,'idx',1);
    vardimID = netcdf.defDim(ncid,'var',nvar);

    % create dimensions
    lat_var_id=netcdf.defVar(ncid, 'lat', 'float', [latdimID]);
    lon_var_id=netcdf.defVar(ncid, 'lon', 'float', [londimID]);
    time_var_id=netcdf.defVar(ncid, 'time', 'float', [timedimID]);
    neuron_var_id = netcdf.defVar(ncid, 'neurons', 'float', [neurondimID]);
    idx_var_id = netcdf.defVar(ncid, 'idx','float',[idxdimID]);
    var_var_id = netcdf.defVar(ncid, 'var','float',[vardimID]);

    %

    % define variables
    modos_id=netcdf.defVar(ncid, 'modos', 'float', [neurondimID,timedimID,vardimID]);
    % netcdf.putAtt(ncid, modos_id, 'units', units);
    %netcdf.putAtt(ncid, modos_tras_id, 'units', units);
    regions_id = netcdf.defVar(ncid, 'regions', 'float',[londimID, latdimID]);
    bmu_id = netcdf.defVar(ncid, 'bmu_map', 'float',[londimID, latdimID]);
    data_id = netcdf.defVar(ncid,'data','float',[timedimID,londimID,latdimID,vardimID]);
    hist_id = netcdf.defVar(ncid,'hist_occur','float',[neurondimID]);
    prob_id = netcdf.defVar(ncid,'prob_occur','float',[neurondimID]);
    qe_id = netcdf.defVar(ncid, 'qe','float',[idxdimID]);
    te_id = netcdf.defVar(ncid, 'te','float',[idxdimID]);
    hist_n_id = netcdf.defVar(ncid,'hist_n_occur','float',[neurondimID]);
    prob_n_id = netcdf.defVar(ncid,'prob_n_occur','float',[neurondimID]);
    qe_n_id = netcdf.defVar(ncid, 'qe_n','float',[idxdimID]);
    te_n_id = netcdf.defVar(ncid, 'te_n','float',[idxdimID]);
    %close definitions
    netcdf.endDef(ncid);

    %put everything in the nc-file:
    netcdf.putVar(ncid, lon_var_id, lon);
    netcdf.putVar(ncid, lat_var_id, lat);
    netcdf.putVar(ncid, time_var_id, time);
    netcdf.putVar(ncid, neuron_var_id, 1:nunits);
    netcdf.putVar(ncid, idx_var_id, 1);
    netcdf.putVar(ncid, var_var_id, 1:nvar);

    netcdf.putVar(ncid, modos_id, modos);
    netcdf.putVar(ncid, regions_id,regions);
    netcdf.putVar(ncid, bmu_id,bmu_map);

    netcdf.putVar(ncid, data_id,data_matrix);
    netcdf.putVar(ncid, hist_id,histo_ocurrencia);
    netcdf.putVar(ncid, prob_id,prob_ocurrencia);
    netcdf.putVar(ncid, qe_id,qe);
    netcdf.putVar(ncid, te_id,te);

    netcdf.putVar(ncid, hist_n_id,histo_ocurrencia_n);
    netcdf.putVar(ncid, prob_n_id,prob_ocurrencia_n);
    netcdf.putVar(ncid, qe_n_id,qe_n);
    netcdf.putVar(ncid, te_n_id,te_n);
    netcdf.close(ncid); 
%                     % %
                    % ii=ii+1;
    end% for n
    end % for ngb_function
end % end for basins


%%