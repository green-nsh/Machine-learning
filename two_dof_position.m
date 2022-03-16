function [func] = two_dof_position(x)
%   �����ɶȻ�е�۷���
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
    r1a=R1+A1*u_1a;     %����1��Aλ��ʸ��

    u_1o=[-0.5*L1;0];
    r1o=R1+A1*u_1o;     %�ο�����ϵλ��(����)

    A2=[cos(a2),-sin(a2);   %����2��ϵλ��
        sin(a2),cos(a2)];
    R2=[r21;r22];
    u_2a=[-0.5*L2;0];
    r2a=R2+A2*u_2a;   %����2��Aλ��ʸ��

    u_2p=[0.5*L2;0];
    r2p=R2+A2*u_2p;     %����2֮P��λ��ʸ��
    %% Լ������1
    fun1=r1a-r2a;   
    %% Լ������2
    fun2=r1o;
    %% ֱ�߹켣
    k=tand(15);
    a0=0;
    b0=0.049;   %���
    p_x=v*t+a0;
    p_y=k*p_x+b0;
    p_=[p_x,p_y].';
    %% Լ������3
    fun3=r2p-p_;

    func=[fun1;fun2;fun3];
end

