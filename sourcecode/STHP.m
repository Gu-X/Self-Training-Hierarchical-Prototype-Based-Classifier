function [Output]=STHP(Input,Mode)
if strcmp(Mode,'ssl')==1
    lambda=Input.Lambda;
    Data_Train=Input.LabeledData;
    Label_Train=Input.GroundTruth;
    LayerNum=Input.LayerNum;
    Output.Syst=[];
    uniquelabel=unique(Label_Train);
    CC=length(uniquelabel);
    Output.Syst.Classes=uniquelabel;
    Output.Syst.LayerNum=LayerNum;
    [~,W]=size(Data_Train);
    Xnorm = sqrt(sum(Data_Train.^2, 2));
    Data  = Data_Train ./ Xnorm(:,ones(1,W));
    or=pi/2;
    for cc=1:1:CC
        seq1=Label_Train==uniquelabel(cc);
        data=Data(seq1,:);L1=length(data(:,1));
        Parm=[];
        for ii=1:1:LayerNum
            R0(ii)=1-cos(or*0.5^(ii-1));
            Parm(ii).IDXC(1).Centre=data(1,:);
            Parm(ii).IDXC(1).Support=1;
            Parm(ii).IDXC(1).NoC=1;
            Parm(ii).IDXC(1).IDX=1;
            Parm(ii).IDXC(1).Radius=R0(ii);
            Parm(ii).IDXC(1).Gmean=data(1,:);
            Parm(ii).IDXC(1).NumD=1;
            Parm(ii).DIDX(1)=1;
            Parm(ii).NumC=1;
        end
        for ii=2:1:L1
            indexG1=0;
            indexG=1;
            for jj=1:1:LayerNum
                if indexG1==0
                    Parm(jj).IDXC(indexG).NumD=Parm(jj).IDXC(indexG).NumD+1;
                    [value,position]=min((pdist2(data(ii,:),Parm(jj).IDXC(indexG).Centre,'euclidean')));
                    value=value.^2;
                    if value>2*Parm(jj).IDXC(indexG).Radius(position) %|| (dist1>max(dist2)) || (dist1<min(dist2))
                        Parm(jj).IDXC(indexG).Centre=[Parm(jj).IDXC(indexG).Centre;data(ii,:)];
                        Parm(jj).IDXC(indexG).NoC=Parm(jj).IDXC(indexG).NoC+1;
                        Parm(jj).IDXC(indexG).Support=[Parm(jj).IDXC(indexG).Support;1];
                        Parm(jj).IDXC(indexG).Radius=[Parm(jj).IDXC(indexG).Radius;R0(jj)];
                        Parm(jj).NumC=Parm(jj).NumC+1;
                        Parm(jj).IDXC(indexG).IDX=[Parm(jj).IDXC(indexG).IDX;Parm(jj).NumC];
                        Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).NumC];
                        indexG1=1;
                    else
                        Parm(jj).IDXC(indexG).Centre(position,:)=Parm(jj).IDXC(indexG).Centre(position,:)*(Parm(jj).IDXC(indexG).Support(position)/(Parm(jj).IDXC(indexG).Support(position)+1))+data(ii,:)/(Parm(jj).IDXC(indexG).Support(position)+1);
                        Parm(jj).IDXC(indexG).Centre(position,:)=Parm(jj).IDXC(indexG).Centre(position,:)./sqrt(sum(Parm(jj).IDXC(indexG).Centre(position,:).^2,2));
                        Parm(jj).IDXC(indexG).Support(position)=Parm(jj).IDXC(indexG).Support(position)+1;
                        Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).IDXC(indexG).IDX(position)];
                        indexG1=0;
                    end
                    indexG=Parm(jj).DIDX(end);
                else
                    Parm(jj).NumC=Parm(jj).NumC+1;
                    Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).NumC];
                    Parm(jj).IDXC(indexG).Centre=data(ii,:);
                    Parm(jj).IDXC(indexG).Support=1;
                    Parm(jj).IDXC(indexG).NoC=1;
                    Parm(jj).IDXC(indexG).IDX=Parm(jj).NumC;
                    Parm(jj).IDXC(indexG).Radius=R0(jj);
                    Parm(jj).IDXC(indexG).Gmean=data(ii,:);
                    Parm(jj).IDXC(indexG).NumD=1;
                    indexG=Parm(jj).NumC;
                    indexG1=1;
                end
            end
        end
        Param1{cc}=Parm;
    end
    %%
    Data_Test=Input.UnlabeledData;
    [L,W]=size(Data_Test);
    data0=Data_Test./repmat(sqrt(sum(Data_Test.^2,2)),1,W);
    L0=1;
    R1=sqrt(2*R0);
    while L0~=L
        L0=L;
        CT={};
        for jj=1:1:LayerNum
            for ii=1:1:CC
                CT{jj,ii}=[];
                for kk=1:1:length(Param1{ii}(jj).IDXC)
                    CT{jj,ii}=[CT{jj,ii};Param1{ii}(jj).IDXC(kk).Centre];
                end
            end
        end
        score=[];
        EstLabData=[];EstLab=[];
        for tt=1:1:LayerNum
            score(tt).conf=zeros(L,CC);
            indic=zeros(L,CC);
            for jj=1:1:L
                for ii=1:1:CC
                    [score(tt).conf(jj,ii),~]=min(pdist2(data0(jj,:),CT{tt,ii}));
                end
            end
            indic(score(tt).conf<=R1(tt))=1;
            indic2=sum(indic,2);
            seq=indic2==1;
            EstLabData=[EstLabData;data0(seq,:)];
            EstLab=[EstLab;vec2ind(indic(seq,:)')'];
            data0(seq,:)=[];
            Data_Test(seq,:)=[];
            L=length(data0(:,1));
            score(1).conf(seq,:)=[];
        end
        scores=score(1).conf;
        scores=exp(-1*(scores).^2);
        [scores1,idx]=sort(scores,2,'descend');
        seq=scores1(:,1)>lambda*scores1(:,2);
        EstLabData=[EstLabData;data0(seq,:)];
        EstLab=[EstLab;idx(seq,1)];
        data0(seq,:)=[];
        L=length(data0(:,1));
        for cc=1:1:CC
            Parm=Param1{cc};
            seq1=EstLab==uniquelabel(cc);
            data=EstLabData(seq1,:);L1=length(data(:,1));
            for ii=1:1:L1
                indexG1=0;
                indexG=1;
                for jj=1:1:LayerNum
                    if indexG1==0
                        Parm(jj).IDXC(indexG).NumD=Parm(jj).IDXC(indexG).NumD+1;
                        [value,position]=min((pdist2(data(ii,:),Parm(jj).IDXC(indexG).Centre,'euclidean')));
                        value=value.^2;
                        if value>2*Parm(jj).IDXC(indexG).Radius(position)
                            Parm(jj).IDXC(indexG).Centre=[Parm(jj).IDXC(indexG).Centre;data(ii,:)];
                            Parm(jj).IDXC(indexG).NoC=Parm(jj).IDXC(indexG).NoC+1;
                            Parm(jj).IDXC(indexG).Support=[Parm(jj).IDXC(indexG).Support;1];
                            Parm(jj).IDXC(indexG).Radius=[Parm(jj).IDXC(indexG).Radius;R0(jj)];
                            Parm(jj).NumC=Parm(jj).NumC+1;
                            Parm(jj).IDXC(indexG).IDX=[Parm(jj).IDXC(indexG).IDX;Parm(jj).NumC];
                            Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).NumC];
                            indexG1=1;
                        else
                            Parm(jj).IDXC(indexG).Centre(position,:)=Parm(jj).IDXC(indexG).Centre(position,:)*(Parm(jj).IDXC(indexG).Support(position)/(Parm(jj).IDXC(indexG).Support(position)+1))+data(ii,:)/(Parm(jj).IDXC(indexG).Support(position)+1);
                            Parm(jj).IDXC(indexG).Centre(position,:)=Parm(jj).IDXC(indexG).Centre(position,:)./sqrt(sum(Parm(jj).IDXC(indexG).Centre(position,:).^2,2));
                            Parm(jj).IDXC(indexG).Support(position)=Parm(jj).IDXC(indexG).Support(position)+1;
                            Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).IDXC(indexG).IDX(position)];
                            indexG1=0;
                        end
                        indexG=Parm(jj).DIDX(end);
                    else
                        Parm(jj).NumC=Parm(jj).NumC+1;
                        Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).NumC];
                        Parm(jj).IDXC(indexG).Centre=data(ii,:);
                        Parm(jj).IDXC(indexG).Support=1;
                        Parm(jj).IDXC(indexG).NoC=1;
                        Parm(jj).IDXC(indexG).IDX=Parm(jj).NumC;
                        Parm(jj).IDXC(indexG).Radius=R0(jj);
                        Parm(jj).IDXC(indexG).Gmean=data(ii,:);
                        Parm(jj).IDXC(indexG).NumD=1;
                        indexG=Parm(jj).NumC;
                        indexG1=1;
                    end
                end
            end
            Param1{cc}=Parm;
        end
    end
    Output.Syst.Param=Param1;
    Output.UnlabeledData=Data_Test;
end
if strcmp(Mode,'tb')==1
    Data_Test=Input.UnlabeledData;
    Syst=Input.Syst;
    LayerNum=Syst.LayerNum;
    CC=length(Syst.Classes);
    data=Data_Test;
    [L,W]=size(data);
    data=data./repmat(sqrt(sum(data.^2,2)),1,W);
    score=zeros(L,CC);
    for jj=1:1:L
        for ii=1:1:CC
            indx=1;
            for kk=1:1:LayerNum
                [a,b]=min(pdist2(data(jj,:),Syst.Param{ii}(kk).IDXC(indx).Centre));
                indx=Syst.Param{ii}(kk).IDXC(indx).IDX(b);
            end
            score(jj,ii)=a;
        end
    end
    [~,q]=max(exp(-1*(score).^2),[],2);
    Output.ConfidenceScores=score;
    Output.Labels=Syst.Classes(q);
end
if strcmp(Mode,'ta')==1
    Data_Test=Input.UnlabeledData;
    Syst=Input.Syst;
    LayerNum=Syst.LayerNum;
    CC=length(Syst.Classes);
    data=Data_Test;
    [L,W]=size(data);
    data=data./repmat(sqrt(sum(data.^2,2)),1,W);
    CT={};
    for ii=1:1:CC
        CT{ii}=[];
        for kk=1:1:length(Syst.Param{ii}(LayerNum).IDXC)
            CT{ii}=[CT{ii};Syst.Param{ii}(LayerNum).IDXC(kk).Centre];
        end
    end
    score=zeros(L,CC);
    for jj=1:1:L
        for ii=1:1:CC
            [score(jj,ii),~]=min(pdist2(data(jj,:),CT{ii}));
        end
    end
    [~,q]=max(exp(-1*(score).^2),[],2);
    Output.ConfidenceScores=score;
    Output.Labels=Syst.Classes(q);
end
end
