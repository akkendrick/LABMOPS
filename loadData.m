% Analysis of the data in Octave
% Assign manually the dimensions:
nx = 512;
ny = 512;
nz = 512;
%%
load velocityCodePrimary_0.001.dat;
%%

vel3 = reshape(velocityCodePrimary_0.001, nx, ny, nz, []);

%%
%imagesc(velNorm(1:end,1:end,125,3));

%%
% load velocityCode.dat
% %%
% 
% nx = 250;
% ny = 250;
% nz = 250;
% 
% vel3 = reshape(velocityCode, nx, ny, nz, 3);
% 
% %%
% test = vel3(:,:,:,1);