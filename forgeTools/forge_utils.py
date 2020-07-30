



#2. Read the Data Tools

def preprocessing_for_das(stream,dx,fmin,fmax,wellHeadChannel=205,wellBottomChannel=1180,trnorm=1,
                          detrend=1,demean=1,taper=1,muteK0=1,muteTopBottom=1,medRemoval=1,bandpass=1):
    import matplotlib.pyplot as plt
    streamout = stream.copy()
    if trnorm == 1:
        streamout.normalize(global_max=False)
    else:
        streamout.normalize(global_max=True)
    if detrend == 1:
        streamout.detrend('linear')
    if demean == 1:
        streamout.detrend('demean')
    if taper == 1:
        streamout.taper(0.02)
    if muteK0 == 1:
        F,K,SPEC = fk(streamout,dx)
        streamout = muteZeroWavenumber(streamout,dx,F,K,SPEC)
    if muteTopBottom == 1:
        for tr in streamout[:wellHeadChannel]:
            tr.data *=0
        for tr in streamout[wellBottomChannel:]:
            tr.data *=0
    if medRemoval == 1:
        streamout = dasMedianRemoval(streamout)
    if bandpass == 1:
        streamout.filter('bandpass',freqmin=fmin,freqmax=fmax,zerophase=True,corners=2)
    return streamout.normalize(global_max=True)

def preprocessing_for_geophones(stream,fmin,fmax):
    import numpy as np
    streamout = stream.select(channel='Z').copy()
    streamout.detrend('linear')
    streamout.detrend('demean')
    streamout.taper(0.02)
    streamout.filter('bandpass',freqmin=fmin,freqmax=fmax,zerophase=True,corners=2)
    gph_kill = [8]
    for tr in gph_kill:
        streamout[tr].data*=np.nan
    return streamout.normalize(global_max=True)

def plotGather(stream,padBefore,padAfter,distanceVector,vmin=-1,vmax=1,trnorm=1):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    dt = stream[0].stats.delta
    nt = stream[0].stats.npts
    if trnorm==True:
        arr = stream2array(stream.normalize(global_max=False))
    else:
        arr = stream2array(stream.normalize(global_max=True))
    plt.imshow(arr,aspect='auto',
               extent=[-padBefore,padAfter,max(distanceVector),min(distanceVector)],
               cmap='viridis',vmin=vmin,vmax=vmax)
    h=plt.colorbar(pad=0.01)
    h.set_label('Normalized Strain-rate',rotation=270,labelpad=20)
    plt.ylabel('Linear Fiber Length [m]')
    plt.xlabel('Time [s]')
    return fig,ax

def plotGeophonesAsGather(stream,padBefore,stations_to_plot='*',color='w',alpha=0.5,rescale=10,shift=0,fig=0):
    import matplotlib.pyplot as plt
    stream.normalize(global_max=False)
    if fig==0:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
    else:
        for tr in stations_to_plot:
            trace = stream[tr].copy()
            plt.plot(trace.times()-padBefore+shift,rescale*trace.data+trace.stats.distance,color=color,alpha=alpha)
    return fig


f

def load_geophone_data_for_event(dataGeophoneDir,eventName,time,padBefore,padAfter):
    #
    # Geophones.Z create raw stream for ploting
    #
    import obspy
    import numpy as np
    velocity_from_geophone = obspy.Stream()
    geophone_locations = np.linspace(645.28,980.56,12)
    dx = geophone_locations[1] - geophone_locations[0] # m

    print('Loading Geophone data for '+str(dataGeophoneDir+eventName))

    for geop in range(12):

        component = '2'
        tr_ = loadSLBGeophone3CChannel(dataGeophoneDir+eventName,time-20,time+20,geop);
        tr = tr_.select(channel=component)[0].copy();
        tr.stats.station = '%02d' % geop;
        tr.stats.distance= geophone_locations[geop];
        velocity_from_geophone += tr;

        component = '1'
        tr_ = loadSLBGeophone3CChannel(dataGeophoneDir+eventName,time-20,time+20,geop);
        tr = tr_.select(channel=component)[0].copy();
        tr.stats.station = '%02d' % geop;
        tr.stats.distance= geophone_locations[geop];
        velocity_from_geophone += tr;

        component = 'Z'
        tr_ = loadSLBGeophone3CChannel(dataGeophoneDir+eventName,time-20,time+20,geop);
        tr = tr_.select(channel=component)[0].copy();
        tr.stats.station = '%02d' % geop;
        tr.stats.distance= geophone_locations[geop];
        velocity_from_geophone += tr;

    velocity_from_geophone.trim(starttime=time-padBefore,endtime=time+padAfter)
    return velocity_from_geophone

def dasMedianRemoval(stream):
    import numpy as np
    A = stream2array(stream)
    Amediantrace = np.median(A,axis=0)
    newstream = stream.copy()
    for tr in newstream:
        tr.data -= Amediantrace
    return newstream

def get_geophone_data(dataGeophoneDir,eventName,eventTime):
    import os
    #establish data directories for event
    try:
        os.mkdir(dataGeophoneDir+eventName)
        print('Creating '+dataGeophoneDir+eventName);
    except:
        print(dataGeophoneDir+eventName+' directory exists already:')
        print(os.listdir(dataGeophoneDir+eventName));

    #download geophone data
    df = getSLBGeophoneFileWgetLines(eventTime-20,eventTime+20)
    print('Downloading %d x SEG-Y Geophone files (15-sec each)from GDR...' % len(df[0]))
    for row in df[0]: # controls download of lines in df[0] to all of 16 hrs
        try:
            os.system(row)
            os.system('mv '+row[-23:]+' '+dataGeophoneDir+eventName)
        except:
            None

def get_das_data(dataDasDir,eventName,eventTime):
    import os
    try:
        os.mkdir(dataDasDir+eventName)
        print('Creating '+dataDasDir+eventName)
    except:
        print(dataDasDir+eventName+' directory exists already:')
        print(os.listdir(dataDasDir+eventName))

    df = getSilixaDASFileWgetLines(eventTime-20,eventTime+20)
    print('Downloading %d Silixa DAS files (15-sec each) from GDR...' % len(df[0]))
    for row in df[0]: # controls download of lines in df[0] to all of 16 hrs
        try:
            os.system(row)
            os.system('mv '+row[-42:]+' '+dataDasDir+eventName)
        except:
            None

def catCreateFromArange(starttime,endtime,duration):
    import pandas as pd
    from datetime import datetime,timedelta
    #
    cat = pd.DataFrame(columns=['starttime','endtime','duration_in_sec'])
    #
    tA = np.arange(starttime,endtime,duration)
    tB = tA[1:]
    tA = tA[:-1]
    cat['starttime']=tA
    cat['endtime']=tB
    cat['duration_in_sec']=duration
    return cat

def getSLBGeophoneFileWgetLines(t0,t1):
    import pandas as pd
    from datetime import datetime
    from obspy import UTCDateTime
    # t0 and t1 are seconds-precision datetime object bounds to retrieve wget lines from get_all_slb.sh file

    #organize DataFrame sorted range by datetime
    struct = pd.read_csv('get_all_slb.sh',skiprows=1,header=None)
    struct[1] = [x[-23:-9] for x in struct[0]]
    struct[2] = [UTCDateTime(datetime.strptime(x,"%Y%m%d%H%M%S")) for x in struct[1]]
    struct = struct[struct[2]>t0]
    struct = struct[struct[2]<=t1]
    struct = struct.sort_values(2)
    return struct

def getSilixaDASFileWgetLines(t0,t1):
    import pandas as pd
    from datetime import datetime
    from obspy import UTCDateTime

    # t0 and t1 are seconds-precision datetime object bounds to retrieve wget lines from get_all_slb.sh file
    # wget -q https://pando-rgw01.chpc.utah.edu/silixa_das_may_03_2019/FORGE_78-32_iDASv3-P11_UTC190503130808.sgy
    #organize DataFrame sorted range by datetime
    struct = pd.read_csv('get_all_silixa.sh',skiprows=1,header=None)
    struct[1] = [x[-16:-4] for x in struct[0]]
    struct[2] = [UTCDateTime(datetime.strptime(x,"%y%m%d%H%M%S")) for x in struct[1]]
#     print(struct[2])
    struct = struct[struct[2]>t0]
    struct = struct[struct[2]<=t1]
    struct = struct.sort_values(2)
    return struct


def getSLBFilenames(dataDir,t0,t1):
    import pandas as pd
    from datetime import datetime
    from obspy import UTCDateTime
    # t0 and t1 are seconds-precision datetime object bounds to retrieve filenames stored in dataDir

    #organize DataFrame sorted range by datetime
    struct = pd.read_csv('get_all_slb.sh',skiprows=1,header=None)
    struct[1] = [x[-23:-9] for x in struct[0]]
    struct[2] = [UTCDateTime(datetime.strptime(x,"%Y%m%d%H%M%S")) for x in struct[1]]
    struct = struct[struct[2]>t0]
    struct = struct[struct[2]<=t1]
    struct = struct.sort_values(2)
    return [dataDir+'/'+x.split('/')[-1] for x in struct[0]]

def loadSLBGeophone3CChannel(dataDir,tA,tB,channel_index):
    import obspy
    import numpy as np
    ## loads 3C data stream for SLB Geophone 12-level channel_index for downloaded data in dataDir
    filenames = getSLBFilenames(dataDir,tA,tB)
    ch = channel_index*3
    S = obspy.Stream()
    for i,f in enumerate(filenames):
        nf = len(filenames)
        st = obspy.read(f,format='segy')
        for ich in np.arange(ch,ch+3,1):
            st[ich].stats.network = '78-32'
            st[ich].stats.station = str(ich)
            st[ich].stats.location = '00'
            if np.mod(ich,3)==0:
                st[ich].stats.channel = '2'
            if np.mod(ich,3)==1:
                st[ich].stats.channel = '1'
            if np.mod(ich,3)==2:
                st[ich].stats.channel = 'Z'
            S+=st[ich]
    S.merge()
    return S

def plotFK(f,k,fk,vmin=-3,vmax=5):
    import numpy as np
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(5,5))
    plt.imshow(np.log10(np.abs(fk)),extent=[min(f),max(f),min(k),max(k)],
               aspect='auto',cmap='jet',vmin=vmin,vmax=vmax)
    h = plt.colorbar()
    h.set_label('Power [strain-rate]')
    plt.xlabel('frequency [1/s]')
    plt.ylabel('wavenumber [1/m]')
    plt.xlim([-80,80])
    plt.ylim([-0.05,0.05])
    c=3000; ax.plot(f,f/c,color='r'); ax.plot(-f,f/c,color='r')
#     c=45; ax.plot(f,f/c,color='r'); ax.plot(-f,f/c,color='r')
    plt.tight_layout()
    return ax

def fkFilter(st,dx,F,K,SPEC):
    import numpy as np
    import matplotlib.pyplot as plt
    dt = st[0].stats.delta
    nk,nf = np.shape(SPEC)

    #eliminate zero freq

    SPEC[int(nk/2),:]*=0

    SPEC[:int(nk/2-20),:]*=0
    SPEC[int(nk/2+20):,:]*=0

    #####
    ifft_arr = np.real(np.fft.ifft2(np.fft.ifftshift(SPEC)))
    st2 = array2stream(ifft_arr,st[0].stats.sampling_rate)
    for tr in st2:
        tr.stats = st[0].stats
    return st2

def muteZeroWavenumber(st,dx,F,K,SPEC):
    import numpy as np
    import matplotlib.pyplot as plt
    dt = st[0].stats.delta
    nk,nf = np.shape(SPEC)

    #eliminate zero freq

    SPEC[int(nk/2-5):int(nk/2+5),:]*=0

#     SPEC[:int(nk/2-20),:]*=0
#     SPEC[int(nk/2+20):,:]*=0

    #####
    ifft_arr = np.real(np.fft.ifft2(np.fft.ifftshift(SPEC)))
    st2 = array2stream(ifft_arr,st[0].stats.sampling_rate)
    for tr in st2:
        tr.stats = st[0].stats
    return st2
