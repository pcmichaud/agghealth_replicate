
clear all 
close all

% workpath='C:\Users\superviseur\Dropbox\flms\data\prix';
% addpath('C:\Users\superviseur\Dropbox\flms\data\prix') 
% workpath2='C:\Users\superviseur\Dropbox\flms\data\oecd\JPE';%aller chercher la base ocde

workpath='/Users/flangot/Dropbox/flms/data/prix';
addpath('/Users/flangot/Dropbox/flms/data/prix') 
workpath2='/Users/flangot/Dropbox/flms/data/oecd/JPE';%aller chercher la base ocde


N=8;%8 countries
year=[1996:2019];
T=length(year);
xsmooth=5;

%%%%%%%%%%%%%%% PRIX
%US, CPI
[AA,BB]=xlsread('DataPriceUS.xlsX','CUSR0000SA0');
tt_us=[1990:1/12:2020+7/12];
xx_us = AA;
%US CPIh
[AA,BB]=xlsread('DataPriceUS.xlsX','CUSR0000SAM');
xx_us = [xx_us AA];
i1=find(tt_us==1996);
i2=find(tt_us==2020);
cpim_us=xx_us(i1:i2-1,:);
t_us=tt_us(i1:i2-1);

%moyenne annuelle à partir de données mensuelles
cpiy_us = monthly2annual(cpim_us);
cpima_us = movmean(cpiy_us,xsmooth);
%figure
%plot(year,cpiy_us,year,cpima_us),grid on

cpi_us=zeros(T,2);
cpi_us(:,1) = cpima_us(:,1)./repmat(cpima_us(1,1),T,1);
cpi_us(:,2) = cpima_us(:,2)./repmat(cpima_us(1,2),T,1);

%figure
%plot(year,cpi_us),grid on,legend('cpi y','cpi health y')

%return

%taux de croissance:
    %lcpi_us=log(cpi_us);
dlcpi_us=log(cpi_us(2:end,:))-log(cpi_us(1:end-1,:));
disp('mean 1996-2019: delta(log(cpi))')
disp('     cpi    cpihealth')
avcpi=mean(dlcpi_us);
disp(avcpi)
disp('phat = mean(cpi health) - mean(cpi)')
phat_us=mean(dlcpi_us(:,2))- mean(dlcpi_us(:,1));
disp(phat_us)

%%%%%%%%%% EUROPE
country=strvcat('DE','DK','FR','IT','NL','SE','SP');
datacpi=zeros(T,N);
datacpih=zeros(T,N);

for jj=1:7
    namepath = strcat(workpath,'/',country(jj,:),'_price');
    eval(['cd ',namepath,';'])
    [AA,BB]=xlsread('CPI.xlsx','FRED Graph');
    tt=[1996:1/12:2020+7/12];
    xx = AA(:,2);
    [AA,BB]=xlsread('CPIH.xlsx','FRED Graph');
    xx = [xx AA(:,2)];
    i1=find(tt==1996);
    i2=find(tt==2020);
    cpim=xx(i1:i2-1,:);
    t=tt(i1:i2-1);
    %moyenne annuelle à partir de données mensuelles
    cpiy = monthly2annual(cpim);
    cpima = movmean(cpiy,xsmooth);
    cpi=zeros(T,2);%base 100 = 1996
    cpi(:,1) = cpima(:,1)./repmat(cpima(1,1),T,1);
    cpi(:,2) = cpima(:,2)./repmat(cpima(1,2),T,1);
    datacpi(:,jj)=cpi(:,1);
    datacpih(:,jj)=cpi(:,2);
end

datacpi(:,8)=cpi_us(:,1);
datacpih(:,8)=cpi_us(:,2);
country2=strvcat('DE','DK','FR','IT','NL','SE','SP','US');

figure
for jj=1:N
    subplot(4,2,jj),plot(year,datacpi(:,jj)*100,year, datacpih(:,jj)*100,'LineWidth',2),grid on,title(country2(jj,:)),axis([1995 2020 1*100 2.2*100])
    if jj==1
        legend('cpi','cpi health','Location','NorthWest')
    end
end

figure
for jj=1:N
    subplot(4,2,jj),plot(year,(datacpih(:,jj)-datacpi(:,jj))*100,'LineWidth',2),grid on,title(country2(jj,:)),axis([1995 2020 -30 50])
    if jj==1
        legend('cpi health - cpi','Location','NorthWest')
    end
end

dlcpi=log(datacpi(2:end,:))-log(datacpi(1:end-1,:));
dlcpih=log(datacpih(2:end,:))-log(datacpih(1:end-1,:));

%cpih_avgrowth =mean(dlcpih);
%cpi_avgrowth =mean(dlcpi);

stop = find(year==2007);
cpih_avgrowth =mean(dlcpih(1:stop,:));
cpi_avgrowth =mean(dlcpi(1:stop,:));

weight_co = [.2827 .0197 .2311 .2117 .0588 .0339 .1617];


disp('  ')
disp('average annual growth rate 1996-2007')
disp('DE  DK  FR  IT  NL  SE  SP  US AvEU')
disp(['cpi_health   ' num2str([100*cpih_avgrowth sum(weight_co.*100.*cpih_avgrowth(1:end-1))])])
disp(['cpi          ' num2str([100*cpi_avgrowth  sum(weight_co.*100.* cpi_avgrowth(1:end-1))])])
disp(['cpi_h - cpi  ' num2str([100*cpih_avgrowth - 100*cpi_avgrowth sum(weight_co.*100.*cpih_avgrowth(1:end-1))-sum(weight_co.*100.*cpi_avgrowth(1:end-1))])])

% vspace = repmat('    ',N,1);
% disp('  ')
% disp('average annual growth rate 1996-2019')
% disp('     cpi_health   cpi  cpi_health-cpi')
% disp([country2 vspace num2str(100*cpih_avgrowth') vspace num2str(100*cpi_avgrowth') ...
%     vspace num2str(100*cpih_avgrowth' - 100*cpi_avgrowth')])

phat = 100*cpih_avgrowth' - 100*cpi_avgrowth';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%charger les données OCDE
eval(['cd ',workpath2,';'])
[AA,BB]=xlsread('data_regFL.xlsx');
country=BB(2:end,1);
eval(['cd ',workpath,';'])

%tothlthcap      double  %10.0g                total health spending per capita (Current prices)
%gdp15ncucap     double  %10.0g                gdp /capita. NCU at 2015 GDP price level
%gdpcap          double  %10.0g                gdp /capita. national currency units
varname=BB(1,2:end);
ipm = find(ismember(varname,'tothlthcap'));
iyreal = find(ismember(varname,'gdp15ncucap'));
iy = find(ismember(varname,'gdpcap'));
%ok
% ipm=7;
% iy = 15; 
% iyreal = 18;

ii1 = find(ismember(country,'Germany'));
ii2 = find(ismember(country,'Denmark'));
ii3 = find(ismember(country,'France'));
ii4 = find(ismember(country,'Italy'));
ii5 = find(ismember(country,'Netherlands'));
ii6 = find(ismember(country,'Sweden'));
ii7 = find(ismember(country,'Spain'));
ii8 = find(ismember(country,'United States'));

%iicountry=[ii1 ii2 ii3 ii4 ii5 ii6 ii7 ii8];
datapm=zeros(T,N);
datay=zeros(T,N);
datayreal=zeros(T,N);


for jj=1:N
    eval(['idata = ii',num2str(jj),';'])
    data = AA(idata,:);
    tt=AA(:,1);debut=find(tt==1996);fin=find(tt==2019);
    datapm(:,jj)=movmean(data(debut:fin,ipm),xsmooth);
    datay(:,jj)=movmean(data(debut:fin,iy),xsmooth);
    datayreal(:,jj)=movmean(data(debut:fin,iyreal),xsmooth);    
end    

share = datapm./datay;
gdpdef0 = datay./datayreal;%GDP deflator 2015
gdpdef  = gdpdef0./repmat(gdpdef0(1,:),T,1);%GDP deflator 1996

datayreal96 = datay./gdpdef;%GDP real, base 1996%%%%%%%%%%%
datapmreal96 = datapm./datacpih;% pm real, base 1996

share_real = datapmreal96./datayreal96;

country2=strvcat('DE','DK','FR','IT','NL','SE','SP','US');

figure
for jj=1:N
    subplot(4,2,jj),plot(year,share(:,jj),year,share_real(:,jj)),grid on, ...
        legend('pm/y','m real / y real'),title(country2(jj,:))
end



dlshare      = log(share(2:end,:))-log(share(1:end-1,:));
dlshare_real = log(share_real(2:end,:))-log(share_real(1:end-1,:));
dlyreal96    = log(datayreal96(2:end,:)) - log(datayreal96(1:end-1,:));
dlmreal96    = log(datapmreal96(2:end,:)) - log(datapmreal96(1:end-1,:));


share_avgrowth      = mean(dlshare);
share_real_avgrowth = mean(dlshare_real);
yreal96_avgrowth    = mean(dlyreal96);
m_avgrowth2         = mean(dlmreal96);

m_avgrowth = 100*share_avgrowth' - phat + 100*yreal96_avgrowth' ;

moy  = [100*mean(share_real,1) sum(weight_co.*100.*mean(share_real(:,1:end-1),1))];
pmoy = [100*mean(share,1)      sum(weight_co.*100.*mean(share(:,1:end-1),1))];
shat = (pmoy-moy)./moy;

disp('  ')
disp('average 1996-2007 (in %)')
disp('DE  DK  FR  IT  NL  SE  SP  US AvEU')
disp(['pm/y   ' num2str([100*mean(share,1)      sum(weight_co.*100.*mean(share(:,1:end-1),1))])])
disp(['m/y    ' num2str([100*mean(share_real,1) sum(weight_co.*100.*mean(share_real(:,1:end-1),1))])])
disp(['shate  ' num2str(100*shat)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cpihc = cpih_avgrowth(1:end-1) + [0.03 0.12 0.09 -0.08 0.2 0.11 0.23]./100;

for tt=1:24
    for ii=1:7
        sharec(tt,ii) = share(tt,ii)/((1+cpihc(ii)-cpi_avgrowth(ii))^(tt-1));
        sharer(tt,ii) = share(tt,ii)/((1+cpih_avgrowth(ii)-cpi_avgrowth(ii))^(tt-1));
    end
end

moyc  = [100*mean(sharec,1) 100*mean(share_real(:,end),1) sum(weight_co.*100.*mean(sharec(:,1:end),1))];
pmoy  = [100*mean(share,1)  sum(weight_co.*100.*mean(share(:,1:end-1),1))];
shatc = (pmoy-moyc)./moyc;


disp('  ')
disp('average 1996-2007 (in %)')
disp('DE  DK  FR  IT  NL  SE  SP  US AvEU')
disp(['CPIc   ' num2str([100*mean(cpihc,1) 100*cpih_avgrowth(end) sum(weight_co.*100.*mean(cpihc(:,1:end),1))])])
disp(['D CPIc ' num2str([100*mean(cpihc-cpi_avgrowth(1:end-1),1) 100*(cpih_avgrowth(end)-cpi_avgrowth(end)) sum(weight_co.*100.*mean(cpihc(:,1:end)-cpi_avgrowth(1:end-1),1))])])
disp(['pm/y   ' num2str([100*mean(share,1)  sum(weight_co.*100.*mean(share(:,1:end-1),1))])])
disp(['m/y    ' num2str([100*mean(sharec,1) 100*mean(share_real(:,end),1) sum(weight_co.*100.*mean(sharec(:,1:end),1))])])
disp(['shate  ' num2str(100*shatc)])


return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


vspace = repmat('    ',N,1);

disp('  ')
disp('average 1996-2019 (in %)')
disp('       pm/y       m/y    ')
disp([country2 vspace num2str(100*mean(share,1)') vspace num2str(100*mean(share_real,1)')])


disp('  ')
disp('average annual growth rate 1996-2019 (in %)')
disp('       pm/y       m/y      p_hat    real_GDP_96  m(deduced)  phat/share ')
disp([country2 vspace num2str(100*share_avgrowth') vspace num2str(100*share_real_avgrowth') vspace num2str(phat) ...
    vspace num2str(100*yreal96_avgrowth') ...
    vspace num2str(m_avgrowth) ...
    vspace num2str(phat./(100*share_avgrowth')) 
    ])

%effet de p, via m
%p baisse => m augmente possiblement




