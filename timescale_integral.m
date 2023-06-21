clf
clear all

addpath('cmap')

set(0,'defaulttextInterpreter','latex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

%%%%%%%%%%%%%
% nhstr = 'No_Hydro_Py_Test_phi_0.34' %'New_No_Hydro_MSD';   
% lsty = ':';
%%%%%%%%%%%%%
% nhstr = 'Py_Test_phi_0.34' %'New_MSD';
% lsty = '-.';
%%%%%%%%%%%%%
% nhstr = 'Py_Test_phi_0.66' %'New_MSD';
% lsty = '-.';
%%%%%%%%%%%%%
nhstr = 'Py_Test_phi_0.02' %'New_MSD';
lsty = '-.';



plateau_data = dlmread(['./Count_Data_Cpp/' nhstr '_N_stats.txt']);
Ls = round(plateau_data(:,1),4);
cols = (dense(length(Ls)));


for kk = 2:length(Ls)
    L_0 = Ls(kk);
    plateau = 2*plateau_data(kk,3); %(plateau_data(kk,2) - plateau_data(kk,1)^2);
    %disp(num2str(plateau))
    %disp(num2str(plateau_data(kk,3) - plateau_data(kk,2)^2))
    
    %     MSD = dlmread(['./Count_Data_Cpp/' nhstr 'mean_BoxL_' num2str(L_0) '.000000_phi_0.34.txt']);
    MSD = dlmread(['./Count_Data_Cpp/' nhstr '_MSDmean_BoxL_' num2str(L_0,'%.6f') '.txt']);
    simTime = 0.5*(1:length(MSD)-1);
    simNC = MSD(2:end);
    %     simErr = dlmread(['./Count_Data_Cpp/' nhstr 'error_BoxL_' num2str(L_0) '.000000_phi_0.34.txt']);
    simErr = dlmread(['./Count_Data_Cpp/' nhstr '_MSDerror_BoxL_' num2str(L_0,'%.6f') '.txt']);
    simErr(1) = [];

    xt = simTime;
    yt = (1-simNC./plateau).^2;
    ep = (1-(simNC+simErr)./plateau).^2;
    em = (1-(simNC-simErr)./plateau).^2;

    %disp(yt(end))

    hh = plot(xt, yt,lsty, 'linewidth',5,'color',[cols(kk,:)],'MarkerFaceColor',[0.8*cols(kk,:)],'markersize',5);
    hold all
    eyu = ep(1:end-1)';
    eyd = em(1:end-1)';
    hfil = fill([xt(1:end-1) flip(xt(1:end-1))],[eyu flip(eyd)],cols(kk,:));
    set(hfil,'facealpha',0.2)
    ylim([1e-7 1.2])
    
    
    %%%%%%% fit tail
    % look at the errbar ratio to get a scaled measume of uncertianty
    ferr = (abs(yt-em)./abs(yt-ep))-1;
    thresh = 0.25;
    idxf = find(ferr > thresh,1);
    if isempty(idxf)
        idxf = round(0.9*length(yt));
    end
    idxs  = idxf-round(0.75*idxf);
    
    %disp(num2str(yt(idxf)))
    if((yt(idxf) > 1e-3) && yt(1) > 1e-1)
        
        Const = polyfit(log(xt(idxs:idxf)),log(yt(idxs:idxf)), 2);
        Xfit = xt(1:idxs);
        dx = Xfit(2)-Xfit(1);
        Yfit = yt(1:idxs)';
        tol = 1e-4;
        spacing = dx;
        while Yfit(end)> tol
            Xnew = Xfit(end)+spacing;
            Ynew = exp(polyval(Const,log(Xnew)));
            Xfit = [Xfit Xnew];
            Yfit = [Yfit Ynew];
            reldiff = 2.0*abs(Yfit(end) - Yfit(end-1))/abs(Yfit(end) + Yfit(end-1));
            if reldiff < 1e-2
                spacing = 2*spacing;
            else
                spacing = 0.5*spacing;
            end
        end
        hold all
        plot(Xfit,Yfit,'-r')
        hold all
        plot(xt(idxs:idxf),yt(idxs:idxf),':','color','b','linewidth',4)
    else
        idxf = find(yt < 5e-6,1);
        Xfit = xt(1:idxf);
        Yfit = yt(1:idxf)';
        xt(idxf+1:end) = [];
        yt(idxf+1:end) = [];
        plot(Xfit,Yfit,'-r')
        hold all
    end
    
   
    
    %%%%%%%%% fit start
    x0 = [0,0,0,0,0,0,0,0];
    fitfun = fittype('1 + a*x.^(1/2)+ b*x + c*x.^(3/2) + d*x.^(2) + f*x.^(5/2) + g*x.^(3) + h*x.^(7/2) + p*x.^(4)','dependent',{'y'},'independent',{'x'},...
    'coefficients',{'a','b','c','d','f','g','h','p'});
    fxt = [1e-8 xt(1:10)];
    fyt = [1 yt(1:10)'];
    
    [fitted_curve,gof] = fit(fxt',fyt',fitfun,'StartPoint',x0);
    coeffvals = coeffvalues(fitted_curve);
    xsmall = logspace(-8,log10(0.5),1000); %[1e-8:1e-3:0.5];
    ysmall = fitted_curve(xsmall);
    hold all
    plot(xsmall,ysmall,'g')
    %%%%%%%%%%%%%%%%%%%
    
    
    Tscale_fit = 2*trapz(Xfit,Yfit)+2*trapz(xsmall,ysmall);
    Tscale_data = 2*trapz(xt,yt);
    rel_err = abs((Tscale_fit-Tscale_data)/(0.5*(Tscale_fit+Tscale_data)));
    disp(num2str([Tscale_fit Tscale_data rel_err]))
    
    Small_int(kk) = trapz(xsmall,ysmall);
    Tscale(kk) = Tscale_fit;
    Tscale_d(kk) = Tscale_data;
    set(gca, 'YScale', 'log', 'XScale', 'log')
end
