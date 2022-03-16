% main_function
clc;
clear;
syms a1 r11 r12 L1
syms a2 r21 r22 L2
syms t v
L10=0.029;  
L20=0.026;
t0=0;
x0=[L10;L20;t0];
func0=two_dof_position(x0);
result0=solve(func0,a1,a2,r11,r12,r21,r22); %结构数组

r11_v=double(result0.r11);
r12_v=double(result0.r12);
r21_v=double(result0.r21);
r22_v=double(result0.r22);
a1_v=double(result0.a1);
a2_v=double(result0.a2);
result0=[a1_v;a2_v;r11_v;r12_v;r21_v;r22_v];

%% 杆件优化
d1=0.02;   
L1_M=L10-d1:d1/5:L10+d1;
L2_M=L20-d1:d1/5:L20+d1;
T_M=0:0.2:1;

% L1_M=[L10-d1,L10+d1];
% L2_M=[L20-d1,L20+d1];
% T_M=[0,1];

 n1=length(L1_M);
 n2=length(T_M);
 
 N=0;   %循环次数
 for i=1:n1
     L1=L1_M(i);
     for j=1:n1
         L2=L2_M(j);
         for k=1:n2
             t=T_M(k);
             x_=[L1,L2,t];
             func=two_dof_position(x_);
             result_=solve(func,a1,a2,r11,r12,r21,r22);
            r11_v=double(result_.r11);
            r12_v=double(result_.r12);
            r21_v=double(result_.r21);
            r22_v=double(result_.r22);
            a1_v=double(result_.a1);
            a2_v=double(result_.a2);
            result_=[a1_v(1);a2_v(1);r11_v(1);r12_v(1);r21_v(1);r22_v(1)];
            
            if isreal(result_)==1
                L1M(i,j,k)=L1;
                L2M(i,j,k)=L2;
                LM(i,j,k)=L1+L2;
            else
                L1M(i,j,k)=NaN;
                L2M(i,j,k)=NaN;
                LM(i,j,k)=NaN;
            end
            clc
            N=N+1;
            progress=roundn(N/(n1*n2*n1)*100,0);
            disp(['Progress: ',num2str(progress),'%'])
         end
     end
 end
 
 %% 搜索最小值
 sz=size(LM);
 [L_min,index]=min(LM(:));      %找出杆长之和的最小值
 [r,c,p]=ind2sub(sz,index);     %找出原来2x2x2索引
 L1_m=L1M(r,c,p);
 L2_m=L2M(r,c,p);
 L_min=L_min;
 
 
 







