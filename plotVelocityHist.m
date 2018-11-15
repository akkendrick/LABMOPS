dir = 'highPressureSim/';

beadFile = strcat(dir,'beadPack.dat');
beadPack = load(beadFile);
beadPack = reshape(beadPack, 101,101,101);

velocityFile = strcat(dir, 'velocity.dat');
velocity = load(velocityFile);
velocity = reshape(velocity', 101,101,101,3);

velocityNormFile = strcat(dir,'velocityNorm.dat');
velocityNorm = load(velocityNormFile);
velocityNorm = reshape(velocityNorm, 101,101,101);

poreSize = load('poreSize.txt');
poreSize = reshape(poreSize, 101,101,101);

beadPack_py = load('beadPack_py.txt');
beadPack_py = reshape(beadPack_py, 101,101,101);

%%
% Plot slice of beadPack
%figure(1)
%slice(beadPack,10,10,10)

% figure(2)
% vel1 = velocity(:,:,:,1);
% slice(vel1,30,20,10)
% 
% figure(3)
% vel1 = velocity(:,:,:,2);
% slice(vel1,30,20,10)
% 
% figure(4)
% vel1 = velocity(:,:,:,3);
% slice(vel1,30,20,10)

%figure(5)
%slice(velocityNorm,30,20,10)

% Attempt to scale velocity data from LB units to 'real' units
length_p = 10^(-3); % [m]
time_p = 1.59; % [s]

del_x = 1/101; % 1/N
del_t = 1/30000; % 1/N_iter

velocityNormScale = (length_p/time_p)*(del_x/del_t).*velocityNorm;
velocityNormScale(beadPack == 1) = NaN;
velocityNormScale(beadPack == 2) = NaN;


figure(6)
slice(velocityNormScale,1,1,1)

% Apply matrix rotations to align python and Matlab data
beadPack_mod = imrotate(beadPack_py,-90);
beadPack_mod = permute(beadPack_mod,[3 2 1]);
beadPack_mod = imrotate(beadPack_mod,-270);
beadPack_mod = permute(beadPack_mod,[2 1 3]);

poreSize_mod = imrotate(poreSize,-90);
poreSize_mod = permute(poreSize_mod,[3 2 1]);
poreSize_mod = imrotate(poreSize_mod,-270);
poreSize_mod = permute(poreSize_mod,[2 1 3]);

poreSize_mod(beadPack_mod == 0) = NaN;
%poreSize_mod(beadPack_mod == 2) = NaN;

%poreSize(beadPack_py == 0) = NaN;


% figure(19)
% slice(beadPack_py,1,1,1)
% 
% figure(20)
% slice(beadPack_mod,1,1,1)

figure(18)
slice(poreSize_mod,1,1,1)

figure(21)
slice(poreSize,1,1,1)




%%
% Calculate mean pore velocity
meanVel = mean(mean(mean(velocityNormScale,'OmitNan'))) %m/s (?)

figure(7)
%subplot(1,1,2)

% Change binning to include 
% check 0 velocity values, where are these located in the matrix

histogram(velocityNormScale)
grid on
box on
title('Bead Pack Velocity Histogram')
xlabel('Velocity Norm')
ylabel('Frequency')
set(gca,'FontSize',14)

figure(22)
histogram(poreSize_mod)


%% Attempt to segment and identify pore size distribution
[BW, N] = bwlabeln(beadPack,8);
RGB = label2rgb(squeeze(BW(1,:,:)));

figure()
slice(RGB,30,20,10)