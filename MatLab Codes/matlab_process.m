%% PD Rest
clear; 
clc

% You will need EEGLab for this script to run: https://sccn.ucsd.edu/eeglab/index.php
% PLEASE locate the standard-10-5-cap385.elp file in your EEGLab and paste the path below
locpath=("C:\eeglab2021.1\plugins\dipfit\standard_BESA\standard-10-5-cap385.elp");
PDsx=[801 802 803 804 805 806 807 808 809 810 811 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829];
CTLsx=[894 908 8010 890 891 892 893 895 896 897 898 899 900 901 902 903 904 905 906 907 909 910 911 912 913 914 8060];
Fs=500;
highPassCutoff=1; % Hz
% Data are 68 chans: 1=63 is EEG, 64 is VEOG, 66-68 is XYZ accelerometer on hand (varied L or R).  
% Ref'd to CPz - - will want to retrieve that during re-referencing.  See below for code for that.
% See bottom of script for the stimulus presentation script used to collect these data
% ON or OFF medication.  CTL only had 1 session
% Note, 8070 only has a little bit of eyes closed data.
% Must have started recording too late.  
% ---------- Get event types
                % All data start with 1 min of eyes closed rest:
                    % trigger 3 happens every 2 seconds
                    % trigger 4 happens every 2 seconds
                % Followed by 1 min of eyes open rest:
                    % trigger 1 happens every 2 seconds
                    % trigger 2 happens every 2 seconds
% Note: 803 S1 is bad.  Don't use.
% controls are >850
PDi=1;
Cri=1;
for subj=[PDsx,CTLsx]
    if subj<850
        isPD=true;
    else
        isPD=false;
    end
    for session=1:2 % ON or OFF medication.  CTL only had 1 session
        if (isPD && session==2) || (~isPD && session==1)  % If not ctl, do session 2
                disp(['Do Rest --- Subno: ',num2str(subj),'      Session: ',num2str(session)]);    
                load([num2str(subj),'_',num2str(session),'_PD_REST.mat'],'EEG');
                data=EEG.data;
                

                % Get EEGs only
                data=data(1:63,:);% 1=63 is EEG
                
                % ---------- Remove CONSISTENLY BAD channels
                data=data([1:4,6:9,11:20,22:26,28:63],:);

                % high pass filter
                parfor chan=1:size(data,1)
                    d=data(chan,:);
                    d=double(d);
                    d=transpose(d);
                    d1=highpass_only(d,Fs,highPassCutoff);
                    data(chan,:)=transpose(d1);
                end
                
                % cut
                datac=data(:,1:Fs*60); % 1st 60 sec- eyes closed?
                %datac=data(:,Fs*60:Fs*60*2);

                %fname
                if isPD
                    fname=strcat('P',num2str(PDi,'%02d'));
                    PDi=PDi+1;
                else
                    fname=strcat('C',num2str(Cri,'%02d'));
                    Cri=Cri+1;
                end
                
                % csv
                csvwrite(strcat(fname,'.csv'), datac);

                % plot
                f=figure("Position",[534,890,1711,420]);
                plot(datac');
                box off
                print(fname, '-dpng', '-r300'); % -r300 specifies 300 DPI
                close(f);
                
        end
    end
end