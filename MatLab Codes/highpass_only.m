function out=highpass_only(data,Fs,highpass_freq)

% design low pass filter to avoid aliasing
hpFilt = designfilt('highpassiir', ...        % Response type
        'PassbandFrequency',highpass_freq+.25, ...     % Frequency constraints
        'StopbandFrequency',highpass_freq, ...
        'PassbandRipple',0.1, ...          % Magnitude constraints
        'StopbandAttenuation',100, ...
        'DesignMethod','ellip', ...      % Design method
        'MatchExactly','passband', ...   % Design method options
        'SampleRate',Fs)  ;             % Sample rate

% fvtool(lpFilt)

% perform low pass filter
NR=Fs*20; % sample length for edge- 2 second
tempdata=data(:);
% for removing edge effect, Prefixing AND Appending the original signal with a small number (10-100) of the beginning AND ending sections of the signal, respectively.
% The sections are flipped and shifted in order to maintain continuity in signal level and slope at the joining points.

% e1=2*tempdata(1)-flipud(tempdata(2:NR+1));
% e2=2*tempdata(end)-flipud(tempdata(end-NR:end-1));

e1=flipud(tempdata(2:NR+1));
e2=flipud(tempdata(end-NR:end-1));

tempdata2=[e1;tempdata;e2];
tempdata2=filtfilt(hpFilt,tempdata2);
%After filtering, the prefixed AND the appended portions of the filtered signal are removed.
tempdata2=tempdata2(NR+1:end-NR);

out=tempdata2;
end