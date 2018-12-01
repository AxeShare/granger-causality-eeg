load sampleEEGdata

chan1name = 'c3';
chan2name = 'f5';

timewin = 200; % in ms
order   =  27; % in ms
times2save = -400:20:1000; % in ms

timewin_points = round(timewin/(1000/EEG.srate));
order_points   = round(order/(1000/EEG.srate));

chan1 = find(strcmpi(chan1name,{EEG.chanlocs.labels}));
chan2 = find(strcmpi(chan2name,{EEG.chanlocs.labels}));

eegdata = bsxfun(@minus,EEG.data([chan1 chan2],:,:),mean(EEG.data([chan1 chan2],:,:),3));
times2saveidx = dsearchn(EEG.times',times2save');


[x2y,y2x] = deal(zeros(1,length(times2save)));
bic = zeros(length(times2save),15);

for timei=1:length(times2save)
   
    tempdata = squeeze(eegdata(:,times2saveidx(timei)-floor(timewin_points/2):times2saveidx(timei)+floor(timewin_points/2)-mod(timewin_points+1,2),:));

    for triali=1:size(tempdata)
        tempdata(1,:,triali) = zscore(detrend(squeeze(tempdata(1,:,triali))));
        tempdata(2,:,triali) = zscore(detrend(squeeze(tempdata(2,:,triali))));
    end
    
    tempdata = reshape(tempdata,2,timewin_points*EEG.trials);
    
    [Ax,Ex] = armorf(tempdata(1,:),EEG.trials,timewin_points,order_points);
    [Ay,Ey] = armorf(tempdata(2,:),EEG.trials,timewin_points,order_points);
    [Axy,E] = armorf(tempdata     ,EEG.trials,timewin_points,order_points);
    

    y2x(timei)=log(Ex/E(1,1));
    x2y(timei)=log(Ey/E(2,2));
    
    for bici=1:size(bic,2)
        [Axy,E] = armorf(tempdata,EEG.trials,timewin_points,bici);
        bic(timei,bici) = log(det(E)) + (log(length(tempdata))*bici*2^2)/length(tempdata);
    end
end

figure
plot(times2save,x2y)
hold on
plot(times2save,y2x,'r')
legend({[ 'GP: ' chan1name ' -> ' chan2name ];[ 'GP: ' chan2name ' -> ' chan1name ]})
title([ 'Window length: ' num2str(timewin) ' ms, order: ' num2str(order) ' ms' ])
xlabel('Time (ms)')
ylabel('Granger prediction estimate')


figure

subplot(121)
plot((1:size(bic,2))*(1000/EEG.srate),mean(bic,1),'--.')
xlabel('Order (converted to ms)')
ylabel('Mean BIC over all time points')

[bestbicVal,bestbicIdx]=min(mean(bic,1));
hold on
plot(bestbicIdx*(1000/EEG.srate),bestbicVal,'mo','markersize',15)

title([ 'Optimal order ' num2str(bestbicIdx) ' (' num2str(bestbicIdx*(1000/EEG.srate)) ' ms)' ])

subplot(122)
[junk,bic_per_timepoint] = min(bic,[],2);
plot(times2save,bic_per_timepoint*(1000/EEG.srate),'--.')
xlabel('Time (ms)')
ylabel('Optimal order (converted to ms)')
title('Optimal order (in ms) at each time point')



d1 = chi2rnd(2,1,1000);
d2 = chi2rnd(2,1,1000);

[y1,x1]=hist(d1,50);
[y2,x2]=hist(d2,50);
[y3,x3]=hist(d1-d2,50);

figure
subplot(221)
plot(x1,y1,'k')
hold on
plot(x2,y2,'r')

subplot(223)
plot(x3,y3)
set(gca,'xlim',[-10 10])

d1 = chi2rnd(7,1,1000);
d2 = chi2rnd(7,1,1000);

[y1,x1]=hist(d1,50);
[y2,x2]=hist(d2,50);
[y3,x3]=hist(d1-d2,50);
