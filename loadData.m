% Analysis of the data in Octave
% Assign manually the dimensions:
nx = 512;
ny = 512;
nz = 512;
%%
load velocityNormCodeSecondary_0.00002.dat;
%%
velNorm = reshape(velocityNormCodeSecondary_0_00002, nx, ny, []);

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