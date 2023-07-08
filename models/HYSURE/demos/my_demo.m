% This is a demo file that exemplifies the use of the HySure algorithm.
% See the file README for more information.
% 
% It corresponds to the example given in [1] using dataset B (Pavia
% University dataset), with the fusion of a simulated hyperspectral image 
% with a simulated multispectral image. 
%  
% See [1] for more details, but essentially we used the original image 
% (with high resolution both in the spatial and in the spectral) as ground 
% truth. To create a hyperspectral image, we spatially blurred the ground  
% truth one, and then downsampled the result by a factor of 4 in each 
% direction. We then filtered it with the Starck-Murtagh filter. To create 
% the panchromatic/multispectral images, the spectral response of the  
% IKONOS satellite was used. Gaussian noise was added to the hyperspectral 
% image (SNR=30 dB) and to the panchromatic/multispectral images
% (SNR=40 dB).
% 
% The downsampling factor and SNR are values that can be modified 
% (see below). 
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A convex formulation for hyperspectral image superresolution via 
%        subspace-based regularization,” IEEE Trans. Geosci. Remote Sens.,
%        to be publised.

% % % % % % % % % % % % % 
% 
% Version: 1
% 
% Can be obtained online from: https://github.com/alfaiate/HySure
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2015 Miguel Simoes, Jose Bioucas-Dias, Luis B. Almeida 
% and Jocelyn Chanussot
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% 
% % % % % % % % % % % % % 
clear; close all;
addpath('../src', '../src/utils');
downsamp_factor = 4; % Downsampling factor

ms_bands = 8;

lambda_R = 1e1;
lambda_B = 1e1;

p = 10;

basis_type = 'VCA';
lambda_phi = 5e-4;
lambda_m   = 1e0;

%LOAD LRHSI
dirname     = 'F:\NNDIFFUSE_FUSION_subset\AVONPM\LRHSI';
LRHSI_fname = 'AVONPM_subset_LRHSI.img';
LRHSI_fname = fullfile(dirname, LRHSI_fname);
dims1       = 400;
dims2       = 320;
dims3       = 360;
LRHSI       = multibandread(LRHSI_fname, [dims1/4 dims2/4 dims3],'uint16',0,'bip','ieee-le');
Yhim        = LRHSI;

%LOAD HRMSI
dirname     = 'F:\NNDIFFUSE_FUSION_subset\AVONPM\HRMSI_wv3';
HRMSI_fname = 'AVONPM_subset_HRMSI_wv3.img';
HRMSI_fname = fullfile(dirname, HRMSI_fname);
HRMSI       = multibandread(HRMSI_fname, [dims1 dims2 8],'uint16',0,'bip','ieee-le');
Ymim = HRMSI;

intersection    = cell(1,length(ms_bands));
intersection{1} = 1:13;
intersection{2} = 10:27;
intersection{3} = 23:44;
intersection{4} = 41:53;
intersection{5} = 49:66;
intersection{6} = 67:76;
intersection{7} = 77:110;
intersection{8} = 94:144;
contiguous      = intersection;

% Blur's support: [hsize_h hsize_w]
hsize_h           = 10;
hsize_w           = 10;
shift             = 1; % the 'phase' parameter in MATLAB's 'upsample' function
blur_center       = 0; % to center the blur kernel according to the simluated data
[V, R_est, B_est] = sen_resp_est(Yhim, Ymim, downsamp_factor, intersection, contiguous, p, lambda_R, lambda_B, hsize_h, hsize_w, shift, blur_center);

Zimhat            = data_fusion(Yhim, Ymim, downsamp_factor, R_est, B_est, p, basis_type, lambda_phi, lambda_m);

%Save Zimhat (right-click on the workspace variables, if not saving is not
%automated)
