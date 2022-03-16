function [func] = two_dof_position(x)
%   二自由度机械臂仿真
    syms a1 r11 r12 L1
    syms a2 r21 r22 L2
    syms t v
    
    L1=x(1);
    L2=x(2);
    t=x(3);
    v=0.0038;
    
    %% 
    R1=[r11;r12];
    A1=[cos(a1),-sin(a1);
        sin(a1),cos(a1)];
    u_1a=[0.5*L1;0];
    r1a=R1+A1*u_1a;     %构件1上A位置矢量

    u_1o=[-0.5*L1;0];
    r1o=R1+A1*u_1o;     %参考坐标系位姿(机架)

    A2=[cos(a2),-sin(a2);   %构件2体系位姿
        sin(a2),cos(a2)];
    R2=[r21;r22];
    u_2a=[-0.5*L2;0];
    r2a=R2+A2*u_2a;   %构件2上A位置矢量

    u_2p=[0.5*L2;0];
    r2p=R2+A2*u_2p;     %构件2之P点位姿矢量
    %% 约束方程1
    fun1=r1a-r2a;   
    %% 约束方程2
    fun2=r1o;
    %% 直线轨迹
    k=tand(15);
    a0=0;
    b0=0.049;   %起点
    p_x=v*t+a0;
    p_y=k*p_x+b0;
    p_=[p_x,p_y].';
    %% 约束方程3
    fun3=r2p-p_;

    func=[fun1;fun2;fun3];
end

