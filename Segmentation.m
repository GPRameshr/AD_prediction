clc
close all
clear all
addpath(genpath('.'))
%% Load dataset
Data='./MRI/';
Data1='./MRI_labels/';
load labels
Data_=dir(Data);
Data_(1:2)=[];
Data1_=dir(Data1);
Data1_(1:2)=[];Features=[];
for N1=1:length(Data1_)
    I=[Data,Data_(N1).name '/'];
    I_=dir(I);
    I_(1:2)=[];
    I2=[Data1,Data1_(N1).name '/'];
    I2_=dir(I2);
    I2_(1:2)=[];
    for N2= 3:3
        q1=[I,I_(N2).name];
        q2=[I2,I2_(N2).name];
        %% Read Images
        [V,imgscan1]=read_image(q1,labels,N1);
        V1=imread(q2);
        imgscan1=adapthisteq(imgscan1);
        imgscan1_=imgscan1;
        figure(2),imshow(imgscan1),title('Ground truth')
        %% Applying Otsu method
        MF=Otsu_TSA(imgscan1);
        B = labeloverlay(imgscan1,imgscan1);
        D = labeloverlay(V,V1);
        for i=1:3
            aa=reshape(MF(i,:),size(imgscan1)).*double(imbinarize(imgscan1));
            aa(aa==0)=[];
            Mean_val(i)=mean(aa);
        end
        [val,id]=max(Mean_val);
        %% Level set
        global dis
        dis=1;
        % Levelset existing
        lsselective_E=Proposed_c(imgscan1_,imgscan1,MF,[id ]);
        % Levelset Proposed
        [lsselective_P]=Proposed(imgscan1_,imgscan1,MF,[id ]);
        % Binarize the segmented image
        Seg_E=imbinarize(lsselective_E);
        Seg_P=(imbinarize(lsselective_P));
        figure,imshow(Seg_P,[]),title('Binarized segmented image')
        Gnd=imbinarize((imgscan1_));
        Gnd1=Gnd;
        Gnd1(Gnd1==0)=-1;
        Gnd1(Gnd1==1)=1.1;
        
        figure(4),imshow(lsselective_P,[]);
        title('Segmented Image')
        figure(5),imshow(B),title('cropped image')
        figure(6),imshow(D),title('Cropped region in original image')
        
        %% Performance measure
        similarity_E = jaccard(logical(Seg_E),logical(((Gnd))));
        Dic_multi_E = dice(logical(Seg_E),logical(((Gnd))));
        [Accuracy_E,~ , ~, ~,~ , Sensitivity_E, ~, Specitivity_E] = EvaluateImageSegmentationScores(logical(Seg_E),logical(((Gnd))));
        similarity_P = jaccard(logical(Seg_P),logical(((Gnd))));
        Dic_multi_P = dice(logical(Seg_P),logical(((Gnd))));
        [Accuracy_P, FN, FP, TP, TN, Sensitivity_P, ~, Specitivity_P] = EvaluateImageSegmentationScores(logical(Seg_P),logical(((Gnd))));
        Result_J=[similarity_E,similarity_P];
        Result_D=[Dic_multi_E,Dic_multi_P];
        Result_ACC=[Accuracy_E,Accuracy_P];
        Result_SEN=[Sensitivity_E,Sensitivity_P];
        Results=[Result_J' Result_D' Result_ACC' Result_SEN'  ];
        Perfomance={'Jacard'  'Dice' 'Accuracy' 'Sensitivity' };
        Images={'Existing','Proposed'};
        array2table(Results,'VariableNames',Perfomance, 'RowNames' ,Images )
        pause(0.1)
        close all
        %% Feature extraction
        opts.gridHist = 1;
        opts.mode = 'nh';
        opts.t = 11;
        %Local Binary pattern
        Local_BP_(N1)=desc_LbeP(Seg_E,opts);
        %Local Directional Pattern
        Local_DP_(N1)=desc_LbeP(Seg_E,opts);
        Features=[Features;Local_BP_,Local_DP_];
    end
end