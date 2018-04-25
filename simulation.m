% load simulated data
load simu_data.mat
V = size(W,1);
LTidx = tril(true(V),-1); % lower triangular index

% high signal-to-noise ratio -----------
K = 5;
gamma = 1.8427; 
fullit = 50;
maxit = 10000;
tol = 1e-5;
Replicates = 10;

rng(0)
[alpha_final1, B_final1, Lambda_final1, F_final1] = SymBilinear(W,y1,K,gamma,fullit,maxit,tol,[],Replicates);

% visualization ----
sig_comps = find(Lambda_final1);
nr = length(sig_comps);

figure
for r=1:nr
    subplot('Position', [(r-1)/nr+1.8/100 0.1 1/nr-2/100 1-0.2]); % [left bottom width height]
    SBL_coef_r = Lambda_final1(sig_comps(r)) * B_final1(:,sig_comps(r)) * B_final1(:,sig_comps(r))';
    SBL_coef_r(1:V+1:end) = 0;
    coefLT = SBL_coef_r(LTidx);
    cmin = min(coefLT);
    cmax = max(coefLT);
    imagesc(SBL_coef_r)
    colormap(b2r(min(cmin,0),max(cmax,0)))
    %colormap(interp1(sample_pts, sample_col, query_pts))
    colorbar
    axis tight
end

% save figure
set(gcf,'Units','Inches','Position',[0,0,12.1,2.75]);
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
saveas(gcf,'SBL_coef_high_snr','pdf')

% low signal-to-noise ratio -----------------
K = 5;
gamma = 9.1062; 
fullit = 250;
maxit = 10000;
tol = 1e-5;
Replicates = 10;

rng(0)
[alpha_final2, B_final2, Lambda_final2, F_final2] = SymBilinear(W,y2,K,gamma,fullit,maxit,tol,[],Replicates);

% visualization ----
sig_comps = find(Lambda_final2);
nr = length(sig_comps);

figure
for r=1:nr
    subplot('Position', [(r-1)/nr+3/100 0.08 1/nr-4/100 1-0.1]); % [left bottom width height]
    SBL_coef_r = Lambda_final2(sig_comps(r)) * B_final2(:,sig_comps(r)) * B_final2(:,sig_comps(r))';
    SBL_coef_r(1:V+1:end) = 0;
    coefLT = SBL_coef_r(LTidx);
    cmin = min(coefLT);
    cmax = max(coefLT);
    imagesc(SBL_coef_r)
    colormap(b2r(min(cmin,0),max(cmax,0)))
    %colormap(interp1(sample_pts, sample_col, query_pts))
    colorbar
    axis tight
end

% save figure
set(gcf,'Units','Inches','Position',[0,0,7.25,3.1]);
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
saveas(gcf,'SBL_coef_low_snr','pdf')
