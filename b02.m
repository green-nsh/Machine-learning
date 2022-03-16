% 粒子群优化算法
% 2022.3.16 by Pucheng 
clc;
clear;
close all;
%% 初始化参数
N=100;      %迭代次数
c_1=2;
c_2=2;
n=2;    %并行计算的粒子数

w_=0.6; %惯性因子
v_max=0.2;

p_best=zeros(1,2);     %局部最优值 全局最优值

x_=[-3,3];      %初始化粒子坐标和速度
v_=[0.05,0.08];

for k=1:n  
    y_(k)=-(x_(k)^3-1)*x_(k);
    p_best(k)=y_(k);
    g_best=p_best(k);       %初始化全局最优值
    if y_(k)>g_best
        g_best=y_(k);
    end
end
%% 粒子群优化过程
X=zeros(N,n);       %位置更新
Y=zeros(N,n);       
V_adjust=zeros(N,n); %限制过的更新速度速度
V=zeros(N,n);       %更新速度
G=zeros(N,1);


for i=1:N
    for m=1:n
        v_(m)=w_*v_(m)+c_1*rand*(p_best(m)-x_(m))+c_2*rand*(g_best-x_(m));
        V(i,m)=v_(m);   %%
        if v_(m)>v_max
           v_(m)=v_max;
        elseif v_(m)<-v_max
           v_(m)=-v_max;
        end
        V_adjust(i,m)=v_(m);    %%
        x_(m)=x_(m)+v_(m);   
        X(i,m)=x_(m);     %%
    end
    
    for k=1:n  
        y_(k)=-(x_(k)^3-1)*x_(k);
        Y(i,k)=y_(k);      %%
        p_best(k)=y_(k);
        if y_(k)>g_best
            g_best=y_(k);
            G(i,1)=g_best; %%
        end
    end   
end

figure;
plot(1:N,V_adjust(:,1),1:N,V_adjust(:,2),'linewidth',2);
ylim([-0.3 0.3]);
grid on;
title('速度更新变化图')

figure;
plot(1:N,Y(:,2),1:N,Y(:,1),'linewidth',2);
text(find(Y(:,2)==g_best),Y(find(Y(:,2)==g_best),2),'o','color','b','HorizontalAlignment','center','FontWeigh','bold');
grid on;
title('局部最优值变化图')

figure;
plot(1:N,G,'linewidth',2);
text(find(Y(:,2)==g_best),Y(find(Y(:,2)==g_best),2),'o','color','r','HorizontalAlignment','center','FontWeigh','bold');
grid on;
title('全局最优值变化图')

% figure;
% plot(1:N,(Y(:,2)+58.6656)/59.136);
% ylim([0 1.5])

